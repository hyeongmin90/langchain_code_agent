from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field

#decompose_into_user_stories
class Epic(BaseModel):
    title: str = Field(description="프로젝트 전체의 최상위 목표 제목")
    goal: str = Field(description="프로젝트를 통해 달성하고자 하는 상세 비즈니스 목표")

class UserStoriesDraft(BaseModel):
    id: str = Field(description="유저 스토리 초안 고유 식별자")
    as_a: str = Field(description="스토리의 주체가 되는 사용자 역할")
    i_want_to: str = Field(description="사용자가 달성하고자 하는 목표나 행동")
    so_that: str = Field(description="그 목표를 통해 얻게 되는 가치나 이유")

class UserStoriesResult(BaseModel):
    epic: Epic = Field(description="프로젝트의 최상위 목표를 정의하는 에픽")
    user_stories_draft: List[UserStoriesDraft] = Field(description="브레인스토밍을 통해 도출된 유저 스토리 초안 목록")


class RefinedUserStoriesDraft(BaseModel):
    id: str = Field(description="유저 스토리 고유 식별자")
    as_a: str = Field(description="스토리의 주체가 되는 사용자 역할")
    i_want_to: str = Field(description="사용자가 달성하고자 하는 목표나 행동")
    so_that: str = Field(description="그 목표를 통해 얻게 되는 가치나 목적")
    priority: str = Field(description="스토리의 중요도 (High, Medium, Low)")

class GroupedUserStories(BaseModel):
    id: str = Field(description="그룹의 고유 식별자")
    epic: str = Field(description="그룹의 최상위 목표를 정의하는 에픽")
    description: str = Field(description="그룹의 설명")
    priority: str = Field(description="그룹의 중요도 (High, Medium, Low)")
    refined_user_stories: List[RefinedUserStoriesDraft] = Field(description="그룹화된 정제된 유저 스토리 목록")

class Dependency(BaseModel):
    from_module: str = Field(description="의존성 출처 모듈 식별자")
    to_module: str = Field(description="의존성 대상 모듈 식별자")
    dependency_type: Literal["needs", "uses", "triggers"] = Field(description="의존성 타입")
    reason: str = Field(description="의존성 이유")

class CrossCuttingConcern(BaseModel):
    name: str = Field(description="횡단 관심사 이름")
    description: str = Field(description="구체적인 구현 방식")
    affected_epics: List[str] = Field(description="영향받는 에픽 (전체 또는 특정)")

class DependenciesResult(BaseModel):
    dependencies: List[Dependency] = Field(description="의존성 목록")

class CrossCuttingConcernsResult(BaseModel):
    cross_cutting_concerns: List[CrossCuttingConcern] = Field(description="횡단 관심사 목록")

class GroupedUserStoriesResult(BaseModel):
    grouped_user_stories: List[GroupedUserStories] = Field(description="기능별로 그룹화된 정제된 유저 스토리 목록")
    group_dependencies: List[Dependency] = Field(description="그룹화된 유저 스토리의 의존성")
    cross_cutting_concerns: List[CrossCuttingConcern] = Field(description="그룹화된 유저 스토리의 횡단 관심사")

class AcceptanceCriteria(BaseModel):
    scenario: str = Field(description="시나리오 제목")
    given: str = Field(description="시나리오가 시작되기 전의 전제 조건")
    when: str = Field(description="사용자가 취하는 특정 행동")
    then: str = Field(description="그 행동으로 인해 발생해야 하는 기대 결과")

class DetailedUserStoriesDraft(BaseModel):
    id: str = Field(description="유저 스토리의 고유 식별자")
    as_a: str = Field(description="스토리의 주체가 되는 사용자 역할")
    i_want_to: str = Field(description="사용자가 원하는 행동이나 기능")
    so_that: str = Field(description="그 행동을 통해 얻는 가치나 목적")
    priority: str = Field(description="스토리의 중요도 (High, Medium, Low)")
    
    description: str = Field(
        description="이 스토리에 대한 설명"
    )
    
    acceptance_criteria: List[AcceptanceCriteria] = Field(
        description="이 스토리가 완료되었음을 증명하는 여러 수용 기준 시나리오 목록"
    )

    business_rules: List[str] = Field(description="이 스토리에 대한 비즈니스 규칙 목록")

class DetailedUserStoriesResult(BaseModel):
    id: str = Field(description="그룹의 고유 식별자")
    epic: str = Field(description="상세화된 유저 스토리의 최상위 목표를 정의하는 에픽")
    description: str = Field(description="상세화된 유저 스토리의 설명")
    priority: str = Field(description="상세화된 유저 스토리의 중요도 (High, Medium, Low)")
    detailed_user_stories: List[DetailedUserStoriesDraft] = Field(description="상세화된 유저 스토리 목록")

class DetailedUserStoriesGroup(BaseModel):
    user_story_group: List[DetailedUserStoriesResult] = Field(description="상세화된 유저 스토리 그룹 목록")

class ScopeAndConstraints(BaseModel):
    scope: List[str] = Field(description="프로젝트의 범위를 정의하는 항목 목록")
    assumptions: List[str] = Field(description="프로젝트의 기반이 되는 가정 목록")
    constraints: List[str] = Field(description="프로젝트의 기술적, 운영적 제약 조건 목록")

class NonFunctionalRequirements(BaseModel):
    non_functional_requirements: List[str] = Field(
        description="프로젝트 전체에 적용되는 비기능 요구사항 목록 (보안, 성능, 안정성 등)"
    )
    scope_and_constraints: ScopeAndConstraints = Field(description="프로젝트 범위, 가정, 제약")

class FinalUserStories(BaseModel):
    id: str = Field(description="그룹의 고유 식별자")
    epic: str = Field(description="그룹화된 유저 스토리의 최상위 목표")
    description: str = Field(description="그룹화된 유저 스토리의 기능 설명")
    priority: str = Field(description="그룹화된 유저 스토리의 중요도 (High, Medium, Low)")
    detailed_user_stories: List[DetailedUserStoriesDraft] = Field(description="그룹화된 유저 스토리 목록")

class ArchitectureResult(BaseModel):
    final_user_stories: List[FinalUserStories] = Field(description="최종 유저 스토리 목록")
    
# Agent States
class DecomposeAgentState(TypedDict):
    user_request: str
    epic: Epic
    raw_user_stories: Optional[UserStoriesResult]
    refined_user_stories: Optional[UserStoriesResult]
    grouped_user_stories: Optional[List[GroupedUserStories]]
    final_result: Optional[DetailedUserStoriesGroup]
    non_functional_requirements: Optional[NonFunctionalRequirements]
    cross_cutting_concerns: Optional[CrossCuttingConcernsResult]
    dependencies: Optional[DependenciesResult]
    

    
