from typing import TypedDict, List, Optional
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

class RefinedUserStoriesResult(BaseModel):
    refined_user_stories: List[List[RefinedUserStoriesDraft]] = Field(description="그룹화된 정제된 유저 스토리 목록")

class AcceptanceCriteria(BaseModel):
    scenario: str = Field(description="시나리오 제목")
    given: str = Field(description="시나리오가 시작되기 전의 전제 조건")
    when: str = Field(description="사용자가 취하는 특정 행동")
    then: str = Field(description="그 행동으로 인해 발생해야 하는 기대 결과")

class FullySpecifiedUserStoriesDraft(BaseModel):
    id: str = Field(description="유저 스토리의 고유 식별자")
    as_a: str = Field(description="스토리의 주체가 되는 사용자 역할")
    i_want_to: str = Field(description="사용자가 원하는 행동이나 기능")
    so_that: str = Field(description="그 행동을 통해 얻는 가치나 목적")
    priority: str = Field(description="스토리의 중요도 (High, Medium, Low)")
    
    detailed_specification: str = Field(
        description="이 스토리를 구현하기 위한 상세 기능 요구사항. 마크다운 형식의 문자열."
    )
    
    acceptance_criteria: List[AcceptanceCriteria] = Field(
        description="이 기능이 완료되었음을 증명하는 여러 수용 기준 시나리오 목록"
    )

class FinalUserStoriesResult(BaseModel):
    final_user_stories: List[FullySpecifiedUserStoriesDraft] = Field(description="최종 유저 스토리 목록")

class ScopeAndConstraints(BaseModel):
    scope: List[str] = Field(description="프로젝트의 범위를 정의하는 항목 목록")
    assumptions: List[str] = Field(description="프로젝트의 기반이 되는 가정 목록")
    constraints: List[str] = Field(description="프로젝트의 기술적, 운영적 제약 조건 목록")

class NonFunctionalRequirements(BaseModel):
    non_functional_requirements: List[str] = Field(
        description="프로젝트 전체에 적용되는 비기능 요구사항 목록 (보안, 성능, 안정성 등)"
    )
    scope_and_constraints: ScopeAndConstraints = Field(description="프로젝트 범위, 가정, 제약")

class AnalysisResult(BaseModel):
    goal: str = Field(description="분석된 고객의 요청사항에 대한 목표")
    functional_requirements: str = Field(description="기능적 요구사항")

class FeedbackAnalysisResult(BaseModel):
    feedback: str = Field(description="작성된 요구사항에 대한 평가")
    is_complete: bool = Field(description="작성된 요구사항이 사용자의 요청사항을 충분히 반영했는지 평가하라. 충분히 반영했다면 true, 아니면 false")

class Dependency(BaseModel):
    to: str = Field(description="의존성 대상")
    dependency_type: str = Field(description="의존성 타입")
    reason: str = Field(description="의존성 설명")

class GroupedEpic(BaseModel):
    group_epic: str = Field(description="그룹화된 유저 스토리의 최상위 목표")
    group_description: str = Field(description="그룹화된 유저 스토리의 기능 설명")
    group_user_stories_id: List[str] = Field(description="그룹화된 유저 스토리 고유 식별자 목록")
    key_entities: List[str] = Field(description="그룹화된 유저 스토리의 핵심 명사(엔티티) 목록")
    actors: List[str] = Field(description="그룹화된 유저 스토리의 주체가 되는 사용자 역할 목록")
    group_dependencies: List[Dependency] = Field(description="그룹화된 유저 스토리의 의존성")
    
class CrossCuttingConcern(BaseModel):
    name: str = Field(description="횡단 관심사 이름")
    description: str = Field(description="구체적인 구현 방식")
    affected_epics: List[str] = Field(description="영향받는 에픽 (전체 또는 특정)")

class ArchitectureResult(BaseModel):
    epic: Epic = Field(description="프로젝트의 최상위 목표를 정의하는 에픽")
    refined_user_stories_grouped: List[GroupedEpic] = Field(description="기능별 분류된 유저 스토리 목록")
    cross_cutting_concerns: List[CrossCuttingConcern] = Field(description="그룹화된 유저 스토리의 횡단 관심사 목록")

# Agent States
class DecomposeAgentState(TypedDict):
    user_request: str
    epic: Epic
    raw_user_stories: Optional[UserStoriesResult]
    refined_user_stories: Optional[UserStoriesResult]
    refined_user_stories_grouped: Optional[List[List[RefinedUserStoriesDraft]]]
    final_specifications: Optional[List[List[FullySpecifiedUserStoriesDraft]]]
    non_functional_requirements: Optional[NonFunctionalRequirements]
    architecture: Optional[ArchitectureResult]

    
