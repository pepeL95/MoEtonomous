#!/usr/bin/env python3
"""
Test script to verify the MoE installation section works correctly.
This script tests all the components mentioned in the installation guide.
"""

import sys
import os
from typing import Dict, List, Tuple


# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

def test_imports() -> Dict[str, bool]:
    """Test all core imports mentioned in the installation guide."""
    results = {}
    
    # Core MoE imports
    try:
        from moe.annotations.core import Expert, MoE, Autonomous
        results["moe.annotations.core"] = True
        print("✅ Core MoE imports successful")
    except ImportError as e:
        results["moe.annotations.core"] = False
        print(f"❌ Core MoE imports failed: {e}")
    
    # LLM imports
    try:
        from dev_tools.enums.llms import LLMs
        results["dev_tools.enums.llms"] = True
        print("✅ LLM imports successful")
    except ImportError as e:
        results["dev_tools.enums.llms"] = False
        print(f"❌ LLM imports failed: {e}")
    
    # Agent imports
    try:
        from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
        results["agents.prebuilt.ephemeral_nlp_agent"] = True
        print("✅ Agent imports successful")
    except ImportError as e:
        results["agents.prebuilt.ephemeral_nlp_agent"] = False
        print(f"❌ Agent imports failed: {e}")
    
    # LangGraph imports
    try:
        import langgraph
        results["langgraph"] = True
        print("✅ LangGraph import successful")
    except ImportError as e:
        results["langgraph"] = False
        print(f"❌ LangGraph import failed: {e}")
    
    # LangChain imports
    try:
        import langchain
        import langchain_core
        import langchain_community
        results["langchain"] = True
        print("✅ LangChain imports successful")
    except ImportError as e:
        results["langchain"] = False
        print(f"❌ LangChain imports failed: {e}")
    
    return results

def test_llm_initialization() -> Tuple[bool, str]:
    """Test LLM initialization with API key."""
    try:
        from dev_tools.enums.llms import LLMs
        
        # Check if API key is set
        if not os.getenv('GOOGLE_API_KEY'):
            return False, "GOOGLE_API_KEY not set in environment"
        
        llm = LLMs.Gemini()
        return True, "LLM initialization successful"
    except Exception as e:
        return False, f"LLM initialization failed: {e}"

def test_agent_creation() -> Tuple[bool, str]:
    """Test agent creation with LLM."""
    try:
        from dev_tools.enums.llms import LLMs
        from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
        
        llm = LLMs.Gemini()
        agent = EphemeralNLPAgent(
            name="TestAgent",
            llm=llm,
            system_prompt="You are a helpful assistant."
        )
        return True, "Agent creation successful"
    except Exception as e:
        return False, f"Agent creation failed: {e}"

def test_optional_dependencies() -> Dict[str, bool]:
    """Test optional dependencies mentioned in the installation guide."""
    results = {}
    
    optional_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("chromadb", "ChromaDB"),
        ("umap", "UMAP"),
        ("sklearn", "Scikit-learn"),
        ("fitz", "PyMuPDF"),
        ("rapidfuzz", "RapidFuzz"),
        ("jira", "JIRA"),
    ]
    
    for import_name, display_name in optional_packages:
        try:
            __import__(import_name)
            results[display_name] = True
            print(f"✅ {display_name} available")
        except ImportError:
            results[display_name] = False
            print(f"⚠️  {display_name} not installed (optional)")
    
    return results

def test_environment_variables() -> Dict[str, bool]:
    """Test if required environment variables are set."""
    results = {}
    
    required_vars = [
        ("GOOGLE_API_KEY", "Google API Key"),
        ("LANGCHAIN_API_KEY", "LangChain API Key"),
    ]
    
    optional_vars = [
        ("OPENAI_API_KEY", "OpenAI API Key"),
        ("SRC", "Source Path"),
        ("VDB_PATH", "Vector Database Path"),
    ]
    
    print("\n🔧 Environment Variables:")
    
    for var_name, display_name in required_vars:
        value = os.getenv(var_name)
        if value:
            results[display_name] = True
            print(f"✅ {display_name}: Set")
        else:
            results[display_name] = False
            print(f"❌ {display_name}: Not set (REQUIRED)")
    
    for var_name, display_name in optional_vars:
        value = os.getenv(var_name)
        if value:
            results[display_name] = True
            print(f"✅ {display_name}: Set")
        else:
            results[display_name] = False
            print(f"⚠️  {display_name}: Not set (optional)")
    
    return results

def main():
    """Run all verification tests."""
    print("🚀 MoE Installation Verification Script")
    print("=" * 50)
    
    # Test imports
    print("\n📦 Testing Imports:")
    import_results = test_imports()
    
    # Test LLM initialization
    print("\n🤖 Testing LLM Initialization:")
    llm_success, llm_message = test_llm_initialization()
    if llm_success:
        print(f"✅ {llm_message}")
    else:
        print(f"❌ {llm_message}")
    
    # Test agent creation
    print("\n👨‍💼 Testing Agent Creation:")
    agent_success, agent_message = test_agent_creation()
    if agent_success:
        print(f"✅ {agent_message}")
    else:
        print(f"❌ {agent_message}")
    
    # Test optional dependencies
    print("\n🔧 Testing Optional Dependencies:")
    optional_results = test_optional_dependencies()
    
    # Test environment variables
    env_results = test_environment_variables()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    # Core functionality
    core_imports_ok = all(import_results.values())
    llm_ok = llm_success
    agent_ok = agent_success
    
    if core_imports_ok and llm_ok and agent_ok:
        print("🎉 CORE FUNCTIONALITY: ✅ WORKING")
        print("   The MoE framework is properly installed and configured!")
    else:
        print("❌ CORE FUNCTIONALITY: ISSUES DETECTED")
        if not core_imports_ok:
            print("   - Import issues detected")
        if not llm_ok:
            print("   - LLM initialization failed")
        if not agent_ok:
            print("   - Agent creation failed")
    
    # Optional features
    optional_installed = sum(optional_results.values())
    total_optional = len(optional_results)
    print(f"\n🔧 OPTIONAL FEATURES: {optional_installed}/{total_optional} installed")
    
    # Environment
    required_env_ok = all([env_results.get("Google API Key", False), 
                          env_results.get("LangChain API Key", False)])
    if required_env_ok:
        print("🌍 ENVIRONMENT: ✅ Properly configured")
    else:
        print("🌍 ENVIRONMENT: ❌ Missing required variables")
    
    print("\n" + "=" * 50)
    
    # Recommendations
    if not core_imports_ok:
        print("💡 RECOMMENDATIONS:")
        print("   - Run: pip install -r requirements.txt")
        print("   - Install missing dependencies")
        print("   - Ensure you're running from the project root directory")
    
    if not llm_ok or not agent_ok:
        print("💡 RECOMMENDATIONS:")
        print("   - Check your .env file configuration")
        print("   - Verify API keys are valid")
        print("   - Install python-dotenv: pip install python-dotenv")
    
    return core_imports_ok and llm_ok and agent_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 