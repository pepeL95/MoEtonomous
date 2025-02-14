from agents.parsers.generic import ArxivXML
from dev_tools.utils.clifont import print_bold, CLIFont


from agents.tools.toolschemas import JiraSchema, ArxivSchema

from langchain_core.tools import tool
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit

import os
from typing import List
from urllib.parse import quote
from jira import JIRA, JIRAError
from urllib.error import URLError
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from requests.exceptions import HTTPError

############################################################# Websearch #########################################################################

class WebsearchToolbox:
    @staticmethod
    def duck_duck_go_tool():
        ddg_search_tool = DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=15))
        ddg_search_tool.name = 'duck_duck_go_tool'
        ddg_search_tool.description = (
            "A wrapper around Duck Duck Go Search. "
            "Useful for when you need to answer questions about current events. "
            "Input to the tool must be a search query, tailor the input query accordingly. "
            "Output is a JSON string array of the query results"
        )
        return ddg_search_tool

################################################################ Jira ###########################################################################

class JiraToolbox:
    jira_wrapper = JiraAPIWrapper()
    _jira = JIRA(server=os.environ['JIRA_INSTANCE_URL'], basic_auth=(os.environ['JIRA_USERNAME'], os.environ['JIRA_API_TOKEN']))
    
    ###############################################################  Jira.Utils ######################################################################
    class Utils:
        def _validate_transition(issue:str, state:str) -> str:
                '''Validates the transition state exists. Returns the transition id when successful'''
                if not state:
                    raise ValueError("No issue key provided. Please provide the issue you want to get transitions for.")
                
                transitions = JiraToolbox._jira.transitions(issue)
                for transition in transitions:
                    if transition['name'].strip().lower() == state.strip().lower():
                        return transition['id']
                
                raise ValueError(
                    f"{state} is not a valid state to transition for issue {issue}. "
                    f"Should be one of {[transition['name'] for transition in transitions]}."
                    )
    
    ##############################################################################################################################################

    @classmethod
    def _get_jql_query_tool(cls):
        toolkit = JiraToolkit.from_jira_api_wrapper(cls.jira_wrapper)
        jql_query_tool = toolkit.get_tools()[0]
        jql_query_tool.name = 'jql_jira_query'
        return jql_query_tool

    @tool(return_direct=False)
    def jql_query_tool(jql_input: str):
        '''This tool is a wrapper around atlassian-python-api's Jira jql API, useful when you need to search for Jira issues.
            The input to this tool is a JQL query string, and will be passed into atlassian-python-api's Jira `jql` function,
            For example, to find all the issues in project "Test" assigned to me, you would pass in the following string:
            project = Test AND assignee = currentUser()
            or to find issues with summaries that contain the word "test", you would pass in the following string:
            summary ~ 'test'''
        
        try:
            tool = JiraToolbox._get_jql_query_tool()
            return tool.invoke(jql_input)
        except ValueError as e:
            return str(e)
        except HTTPError as e:
            return e.response.text
        except Exception as e:
            return 'Something went wrong while trying to fetch issues: ' + str(e)
    
    @tool(return_direct=False, args_schema=JiraSchema.CreateIssue)
    def create_jira_issue(project_key:str, summary:str='', description:str='', assignee:str='Unassigned', issue_type:str='Task'):
        '''Use this tool whenever you need to create a new jira issue or ticket.'''
        try:
            # Sanity checks...
            if not project_key:
                raise ValueError("Please provide a project key for creating the issue")

            issue_dict = {
                'project': {'key': project_key},
                'summary': summary,
                'description': description,
                'assignee': {'name': assignee},
                'issuetype': {'name': issue_type},
            }
            
            new_issue = JiraToolbox._jira.create_issue(fields=issue_dict)
            print_bold(f'{CLIFont.blue}Issue: {new_issue}{CLIFont.reset}')
            return new_issue
        
        except JIRAError as e:
            return e.response.text
        except ValueError as e:
            return str(e)
        except Exception as e:
            return 'Something went wrong while trying to create the issue: ' + str(e)
        
    @tool(return_direct=False, args_schema=JiraSchema.UpdateIssue)
    def update_jira_issue(issue_key:str, summary:str=None, assignee:str=None, description:str=None, duedate:str=None):
        '''Use this tool whenever you need to update any given field from an existing jira issue. For example, whenever you need to update the 'summary' of a given issue.'''            
        try:
            # Sanity checks...
            if not issue_key:
                raise ValueError('Please provide the issue key of the item you want me to update')

            # Fields to update
            fields_to_update = {}
            if summary:
                fields_to_update['summary'] = summary
            if assignee:
                fields_to_update['assignee'] = {'name': assignee}
            if description:
                fields_to_update['description'] = description
            if duedate:
                fields_to_update['duedate'] = duedate
            # If nah' siquiera, sale pa afuera
            if not fields_to_update.values():
                return 'No fields were given to update. Please provide the fields you want me to update'
            
            issue = JiraToolbox._jira.issue(issue_key)
            issue.update(fields=fields_to_update)
            return f'Successfully updated issue {issue_key}'
        
        except JIRAError as e:
            return e.response.text
        except ValueError as e:
            return str(e)
        except Exception as e:
            return 'Something went wrong while trying to update issue: ' + str(e)

    @tool(return_direct=False, args_schema=JiraSchema.TransitionIssueState)
    def transition_issue_state(issue_key:str, state:str=None):
        '''Use this tool when you need to transition a state of an existing issue. For example, transitioning an issue from "To Do" to "Done"'''
        try:
            # Sanity checks...
            if not issue_key:
                raise ValueError('Please provide the issue key of the item you want to transition')
            
            # Fetch issue
            issue = JiraToolbox._jira.issue(issue_key)
            # Validate state is indeed an accepted transition for the given issue
            JiraToolbox.Utils._validate_transition(issue, state)
            # Perform transition
            JiraToolbox._jira.transition_issue(issue, state)
            return f'Successfully transitioned issue {issue_key} to {state}'
        
        except JIRAError as e:
            return e.response.text
        except ValueError as e:
            return str(e)
        except Exception as e:
            return 'Something went wrong while trying to transition states: ' + str(e)


    @tool(return_direct=False, args_schema=JiraSchema.DeleteIssue)
    def delete_jira_issue(issue_key:str):
        '''Use this tool whenever you need to delete an existing jira issue.'''
        try:
            # Sanity checks...
            if not issue_key:
                raise ValueError('Please provide the issue key of the item you want to transition')
            
            issue = JiraToolbox._jira.issue(issue_key)
            issue.delete()
        
        except JIRAError as e:
            return e.response.text
        except ValueError as e:
            return str(e)
        except Exception as e:
            return 'Something went wrong while trying to delete issue: ' + str(e)
    
############################################################# Arxiv #########################################################################

class ArxivToolbox:
    @tool(args_schema=ArxivSchema.ApiSearchItems)
    def build_query_tool(query: str, cat: str, N: int) -> str:
        '''Use this tool when a query json object is given. An example of a query object is: {"query": "space physics", "cat": "physics.space-ph", "N": 10}'''
        
        # Clean out query
        query = quote(query).strip()
        cat = cat.strip()
        
        # Build the api url
        if not query or "recent" in query:
            return f"https://export.arxiv.org/api/query?search_query={cat}&sortBy=submittedDate&sortOrder=descending&max_results={N}"
        
        return f"https://export.arxiv.org/api/query?search_query=cat:{cat}+AND+ti:{query}&sortBy=submittedDate&sortOrder=descending&max_results={N}"

    @tool(return_direct=True)
    def execute_query_tool(url:str) -> List[dict]:
        '''This tool executes a valid query for fetching papers in the Arxiv api system. An example of the ```url``` input is: "https://export.arxiv.org/api/query?search_query=cat:cs.CL+AND+ti:graph%20rag&sortBy=submittedDate&sortOrder=descending&max_results=10"'''
        
        try:
            # Fetch data from the URL
            response = urlopen(url)
            xml_data = response.read().decode('utf-8')
            
            # Parse XML data to dictionary
            articles = ArxivXML.to_dict(xml_data)
            return articles

        except URLError as e:
            raise ConnectionError(f"Failed to fetch data from URL: {e.reason}")

        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML data: {str(e)}")
