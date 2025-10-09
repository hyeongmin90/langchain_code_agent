from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field

class Epic(BaseModel):
    title: str = Field(description="프로젝트 전체의 최상위 목표 제목")
    goal: str = Field(description="프로젝트를 통해 달성하고자 하는 상세 비즈니스 목표")

class UserStoriesDraft(BaseModel):
    id: str = Field(description="사용자 스토리 초안 고유 식별자")
    as_a: str = Field(description="스토리의 주체가 되는 사용자 역할")
    i_want_to: str = Field(description="사용자가 달성하고자 하는 목표나 행동")
    so_that: str = Field(description="그 목표를 통해 얻게 되는 가치나 이유")

class UserStoriesResult(BaseModel):
    epic: Epic = Field(description="프로젝트의 최상위 목표를 정의하는 에픽")
    user_stories_draft: List[UserStoriesDraft] = Field(description="브레인스토밍을 통해 도출된 사용자 스토리 초안 목록")

class AcceptanceCriteria(BaseModel):
    scenario: str = Field(description="시나리오 제목")
    given: str = Field(description="시나리오가 시작되기 전의 전제 조건")
    when: str = Field(description="사용자가 취하는 특정 행동")
    then: str = Field(description="그 행동으로 인해 발생해야 하는 기대 결과")

class FinalUserStoriesResult(BaseModel):
    epic: Epic = Field(description="프로젝트의 최상위 목표를 정의하는 에픽")
    refined_user_stories: List[UserStoriesDraft] = Field(description="정제된 사용자 스토리 목록")
    acceptance_criteria: List[AcceptanceCriteria] = Field(description="각 사용자 스토리에 대한 수용 기준")

class AnalysisResult(BaseModel):
    goal: str = Field(description="분석된 고객의 요청사항에 대한 목표")
    functional_requirements: str = Field(description="기능적 요구사항")

class FeedbackAnalysisResult(BaseModel):
    feedback: str = Field(description="작성된 요구사항에 대한 평가")
    is_complete: bool = Field(description="작성된 요구사항이 사용자의 요청사항을 충분히 반영했는지 평가하라. 충분히 반영했다면 true, 아니면 false")

# Agent States
class DecomposeAgentState(TypedDict):
    user_request: str
    raw_user_stories: Optional[UserStoriesResult]
    refined_user_stories: Optional[UserStoriesResult]
    final_specifications: Optional[FinalUserStoriesResult]

class AnalysisAgentState(TypedDict):
    user_stories: FinalUserStoriesResult
    functional_requirements: str
    is_complete: bool
    feedback: Optional[str]
