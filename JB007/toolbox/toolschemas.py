from pydantic import BaseModel, Field

class ToolSchemas:
    class Jira:
        class CreateIssue(BaseModel):
            project_key: str = Field(description="This is the id for the project to which the new issue will be created.")
            summary: str = Field(description="This is the summary for the given jira issue.", default='')
            assignee: str = Field(description="This is the assignee that the issue will be assigned to", default='Unassigned')
            description: str = Field(description="This is the description that the issue will contain.", default='')
            issue_type: str = Field(description="This is the issue type for the given jira issue", default='Task')

        class UpdateIssue(BaseModel):
            issue_key: str = Field(description="This is the id for the issue that will be updated.")
            summary: str = Field(description="This is the updated/edited summary for the given jira issue.", default=None)
            assignee: str = Field(description="This is the new assignee that the issue will be assigned to", default=None)
            description: str = Field(description="This is the updated description that the issue will contain.", default=None)
            duedate: str = Field(description="This is the updated due date for the given jira issue", default=None)
        
        class TransitionIssueState(BaseModel):
            issue_key: str = Field(description="This is the id for the issue that will be updated.")
            state: str = Field(description="This is the state to transition the jira issue to (e.g. To Do, Done, In Progress)", default=None)

        class DeleteIssue(BaseModel):
            issue_key: str = Field(description="This is the id for the issue that will be deleted.")
        
        class JQL(BaseModel):
            jql_query_str: str = Field(description="This is the rquired JQL query string for querying jira issues.")

    class Arxiv:
        class ApiSearchItems(BaseModel):
            query: str = Field(description="this is the input search query", default="")
            cat: str = Field(description="this is the category taxonomy (convert it to the compatible category taxonomy for the arxiv api)", default="cs.CL")
            N: int = Field(description="this is the number of articles requested", default=10)