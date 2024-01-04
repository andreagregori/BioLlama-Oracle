from pydantic import BaseModel, Field, field_validator
from typing import List


class QueryPubMed(BaseModel):
    query: str = Field(description="This is one possible PubMed query")
    description: str = Field(description="This is a brief description on the query")

    @property
    def query(self):
        return self.query

    @property
    def description(self):
        return self.description


