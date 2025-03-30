import os
import time

from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
from dev_tools.enums.llms import LLMs
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

# Custom exception to signal rate limit issues.
class RateLimitError(Exception):
    pass

class APIKey:
    """
    A class representing an API key with rate-limiting counters and optional persistence of usage across sessions.
    Rate limits are enforced by partitioning time windows and ensuring minimum wait times between requests.
    """
    def __init__(self, name, key, rpm, tpm, rpd, persist_file=None):
        self.name = name
        self.key = key
        
        # If a quota is missing (None), treat it as unlimited.
        self.rpm_limit = rpm if rpm is not None else float('inf')
        self.tpm_limit = tpm if tpm is not None else float('inf')
        self.rpd_limit = rpd if rpd is not None else float('inf')
        
        # Usage counters
        self.rpm_used = 0
        self.tpm_used = 0
        self.rpd_used = 0
        
        # Reset times for each quota window
        self.rpm_reset_time = time.time()
        self.tpm_reset_time = time.time()
        self.rpd_reset_time = time.time()

        # Last call timestamps
        self.last_rpm_call = time.time()
        self.last_tpm_call = time.time()

        # Calculate time partitions
        self.rpm_partition = 60.0 / self.rpm_limit if self.rpm_limit != float('inf') else 0
        self.tpm_partition = 60.0 / self.tpm_limit if self.tpm_limit != float('inf') else 0

        # Optional file path where usage data is stored.
        self.persist_file = persist_file

        # If a persist file is provided, attempt to load usage from it.
        if self.persist_file:
            self._load_persistent_usage()

    def _time_until_next_allowed(self, tokens=1):
        """Calculate time until next allowed call based on RPM and TPM partitions"""
        now = time.time()
        
        # Time needed for RPM partition
        time_since_last_rpm = now - self.last_rpm_call
        rpm_wait = max(0, self.rpm_partition - time_since_last_rpm)
        
        # Time needed for TPM partition (scaled by token count)
        time_since_last_tpm = now - self.last_tpm_call
        tpm_wait = max(0, (self.tpm_partition * tokens) - time_since_last_tpm)
        
        return max(rpm_wait, tpm_wait)

    def can_use(self, tokens=1):
        now = time.time()
        
        # Reset RPM counter if the window has expired
        if now >= self.rpm_reset_time:
            self.rpm_used = 0
            self.rpm_reset_time = now + 60.0
            self.last_rpm_call = now - self.rpm_partition  # Reset last call time
            
        # Reset TPM counter if needed
        if now >= self.tpm_reset_time:
            self.tpm_used = 0
            self.tpm_reset_time = now + 60.0
            self.last_tpm_call = now - self.tpm_partition  # Reset last call time
            
        # Reset RPD counter if needed
        if now >= self.rpd_reset_time:
            self.rpd_used = 0
            self.rpd_reset_time = now + 86400.0

        # Check daily limit first
        if self.rpd_used >= self.rpd_limit:
            return False

        # Check if enough time has elapsed since last call
        wait_time = self._time_until_next_allowed(tokens)
        if wait_time > 0:
            return False

        # Check if adding these tokens would exceed limits
        if self.rpm_used >= self.rpm_limit:
            return False
        if self.tpm_used + tokens > self.tpm_limit:
            return False

        return True

    def record_call(self, tokens=1):
        now = time.time()
        self.rpm_used += 1
        self.tpm_used += tokens
        self.rpd_used += 1
        self.last_rpm_call = now
        self.last_tpm_call = now
        if self.persist_file:
            self._save_persistent_usage()

    def __repr__(self):
        return f"APIKey({self.name})"

    def _load_persistent_usage(self):
        """
        Loads usage counters and reset times from a JSON file.
        """
        import json, os
        if not os.path.exists(self.persist_file):
            return  # No saved usage yet
        try:
            with open(self.persist_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return  # If file is corrupted, ignore

        # The data should be a dict keyed by the API key string.
        usage_info = data.get(self.key, None)
        if usage_info is not None:
            self.rpm_used = usage_info.get('rpm_used', 0)
            self.tpm_used = usage_info.get('tpm_used', 0)
            self.rpd_used = usage_info.get('rpd_used', 0)
            self.rpm_reset_time = usage_info.get('rpm_reset_time', time.time() + 60.0)
            self.tpm_reset_time = usage_info.get('tpm_reset_time', time.time() + 60.0)
            self.rpd_reset_time = usage_info.get('rpd_reset_time', time.time() + 86400.0)
            self.last_rpm_call = usage_info.get('last_rpm_call', time.time())
            self.last_tpm_call = usage_info.get('last_tpm_call', time.time())

    def _save_persistent_usage(self):
        """
        Saves usage counters and reset times to a JSON file.
        """
        if not self.persist_file:
            return  # No persistence enabled

        import json, os
        # Load existing data if any
        if os.path.exists(self.persist_file):
            try:
                with open(self.persist_file, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}

        data[self.key] = {
            'rpm_used': self.rpm_used,
            'tpm_used': self.tpm_used,
            'rpd_used': self.rpd_used,
            'rpm_reset_time': self.rpm_reset_time,
            'tpm_reset_time': self.tpm_reset_time,
            'rpd_reset_time': self.rpd_reset_time,
            'last_rpm_call': self.last_rpm_call,
            'last_tpm_call': self.last_tpm_call
        }

        try:
            with open(self.persist_file, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # If we fail to save, ignore


class KeyRotator:
    def __init__(self, api_keys):
        self.keys = api_keys  # List of APIKey instances.
        self.index = 0

    def get_key(self, tokens=1):
        # Try each key once; if none are available, raise an error.
        for _ in range(len(self.keys)):
            key_obj = self.keys[self.index]
            # Advance the index (round-robin).
            self.index = (self.index + 1) % len(self.keys)
            # Check if the key can be used with the given token cost.
            if key_obj.can_use(tokens=tokens):
                return key_obj
        # All keys are exhausted.
        raise RateLimitError("No available API key at the moment.")


class KeyRotatorAgent:
    def __init__(self, keys, agent):
        self.keys = keys
        self.key_rotator = KeyRotator(self.keys)
        self.agent = agent
        self.key_obj = self.key_rotator.get_key()

    def invoke(self, agent_input, tokens=1, max_retries=3):
        attempts = 0
                
        while attempts < max_retries:
            try:
                wait_time = self.key_obj._time_until_next_allowed(tokens)
                
                if wait_time > 0:
                    print(f"Waiting {wait_time:.1f} seconds before next call...")
                    time.sleep(wait_time)
                
                self.key_obj.record_call(tokens=tokens)
                response = self.agent.invoke(agent_input)
                
                return response

            except RateLimitError:
                raise Exception("All API keys are currently exhausted. Please wait until a quota resets.")
            except Exception as e:
                error_str = str(e)
                if "429 You exceeded your current quota" in error_str:
                    print(
                        "Key is out of sync with the actual API usage. Checking whether daily usage is maxed "
                        "or if we should wait for a minute reset."
                    )
                    # If the daily usage is exceeded, rotate keys
                    if self.key_obj.rpd_used >= self.key_obj.rpd_limit:
                        print("Daily limit reached. Rotating to a different key...")
                        
                        self.key_obj = self.key_rotator.get_key(tokens=tokens)
                        os.environ['GOOGLE_API_KEY'] = self.key_obj.key
                        self.agent.google_api_key = self.key_obj.key
                    else:
                        # Calculate wait time based on partitions
                        wait_time = self.key_obj._time_until_next_allowed(tokens)
                        print(f"Waiting {wait_time:.1f} seconds before next attempt...")
                        time.sleep(wait_time)
                else:
                    print(f"Error invoking agent (attempt {attempts + 1}/{max_retries}): {e}")
                    attempts += 1

        raise Exception(f"Invocation failed after {max_retries} retries.")
    

class EmbeddingsRotatorAgent:
    def __init__(self, keys, agent):
        self.keys = keys
        self.key_rotator = KeyRotator(self.keys)
        self.agent = agent
        self.key_obj = self.key_rotator.get_key()

    def invoke(self, agent_input, tokens=1, max_retries=3):
        attempts = 0
                
        while attempts < max_retries:
            try:
                wait_time = self.key_obj._time_until_next_allowed(tokens)
                
                if wait_time > 0:
                    print(f"Waiting {wait_time:.1f} seconds before next call...")
                    time.sleep(wait_time)
                
                self.key_obj.record_call(tokens=tokens)
                response = self.agent.embed_documents(agent_input)
                
                return response

            except RateLimitError:
                raise Exception("All API keys are currently exhausted. Please wait until a quota resets.")
            except Exception as e:
                error_str = str(e)
                if "429 Quota exceeded" in error_str:
                    print(
                        "Key is out of sync with the actual API usage. Checking whether daily usage is maxed "
                        "or if we should wait for a minute reset."
                    )
                    # If the daily usage is exceeded, rotate keys
                    if self.key_obj.rpd_used >= self.key_obj.rpd_limit:
                        print("Daily limit reached. Rotating to a different key...")
                        
                        self.key_obj = self.key_rotator.get_key(tokens=tokens)
                        os.environ['GOOGLE_API_KEY'] = self.key_obj.key
                        self.agent.google_api_key = self.key_obj.key
                    else:
                        # Calculate wait time based on partitions
                        wait_time = self.key_obj._time_until_next_allowed(tokens)
                        print(f"Waiting {wait_time:.1f} seconds before next attempt...")
                        time.sleep(wait_time)
                else:
                    print(f"Error invoking agent (attempt {attempts + 1}/{max_retries}): {e}")
                    attempts += 1

        raise Exception(f"Invocation failed after {max_retries} retries.")