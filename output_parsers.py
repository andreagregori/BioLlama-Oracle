from pydantic import BaseModel, Field, field_validator
import re


class QueryPubMed(BaseModel):
    query: str = Field(description="This is one possible PubMed query")
    description: str = Field(description="This is a brief description on the query")

    @property
    def query(self):
        return self.query

    @property
    def description(self):
        return self.description


def get_list_from_text(text: str):
    pattern = re.compile(r'\d+\.\s+(.*?)\n')    # regular expression pattern to match list items
    matches = pattern.findall(text)
    return matches


