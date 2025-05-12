"""
Pipeline orchestration with strong typing for the Databricks MLOps framework.
"""
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

from pydantic import BaseModel, Field

from databricks_mlops.core.pipeline import (DataPipeline, DeploymentPipeline,
                                           FeaturePipeline, MonitoringPipeline,
                                           Pipeline, TrainingPipeline)
from databricks_mlops.models.base import Result, StatusEnum
from databricks_mlops.models.config import PipelineConfig
from databricks_mlops.utils.logging import setup_logger

# Type variables for orchestration
T = TypeVar('T', bound=PipelineConfig)
R = TypeVar('R', bound=Result)
P = TypeVar('P', bound=Pipeline)

# Set up logger
logger = setup_logger("orchestrator")


class OrchestrationError(Exception):
    """Exception for errors during pipeline orchestration."""
    pass


class PipelineType(str, Enum):
    """Types of pipelines that can be orchestrated."""
    DATA = "data"
    FEATURE = "feature"
    TRAINING = "training"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    CUSTOM = "custom"


@dataclass
class PipelineStage:
    """Strongly-typed stage definition for pipeline orchestration."""
    name: str
    pipeline_type: PipelineType
    pipeline_class: Type[Pipeline]
    config: PipelineConfig
    depends_on: List[str] = Field(default_factory=list)
    enabled: bool = True
    retry_count: int = 0
    max_retries: int = 3
    on_failure: str = "stop"  # "stop", "continue", "retry"
    timeout_minutes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the stage to a dictionary."""
        return asdict(self)


class OrchestratorResult(Result):
    """Strongly-typed result of pipeline orchestration."""
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    stage_results: Dict[str, Result] = Field(default_factory=dict)
    failed_stages: List[str] = Field(default_factory=list)
    skipped_stages: List[str] = Field(default_factory=list)
    execution_time_seconds: Optional[float] = None


def pipeline_error_handler(func: Callable) -> Callable:
    """
    Decorator for handling errors in pipeline execution.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with error handling
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """Wrapper function with error handling."""
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in pipeline execution: {str(e)}")
            return Result(
                status=StatusEnum.FAILED,
                message=f"Pipeline execution failed: {str(e)}",
                errors=[{"type": type(e).__name__, "message": str(e)}]
            )
    return wrapper


class PipelineOrchestrator:
    """
    Orchestrates the execution of MLOps pipelines with dependencies.
    
    This class manages the execution order of different pipeline stages,
    handles dependencies, and provides error recovery options.
    """
    
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.stages: Dict[str, PipelineStage] = {}
        self.stage_results: Dict[str, Result] = {}
        self.logger = logger
    
    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a pipeline stage to the orchestrator.
        
        Args:
            stage: The pipeline stage to add
            
        Raises:
            OrchestrationError: If a stage with the same name already exists
        """
        if stage.name in self.stages:
            raise OrchestrationError(f"Stage with name '{stage.name}' already exists")
        
        self.stages[stage.name] = stage
        self.logger.info(f"Added stage '{stage.name}' of type {stage.pipeline_type}")
    
    def add_data_stage(
        self,
        name: str,
        config: PipelineConfig,
        depends_on: Optional[List[str]] = None,
        enabled: bool = True,
        **kwargs
    ) -> None:
        """
        Add a data pipeline stage to the orchestrator.
        
        Args:
            name: Name of the stage
            config: Pipeline configuration
            depends_on: Optional list of stage names this stage depends on
            enabled: Whether the stage is enabled
            **kwargs: Additional stage parameters
        """
        stage = PipelineStage(
            name=name,
            pipeline_type=PipelineType.DATA,
            pipeline_class=DataPipeline,
            config=config,
            depends_on=depends_on or [],
            enabled=enabled,
            **kwargs
        )
        self.add_stage(stage)
    
    def add_feature_stage(
        self,
        name: str,
        config: PipelineConfig,
        depends_on: Optional[List[str]] = None,
        enabled: bool = True,
        **kwargs
    ) -> None:
        """
        Add a feature pipeline stage to the orchestrator.
        
        Args:
            name: Name of the stage
            config: Pipeline configuration
            depends_on: Optional list of stage names this stage depends on
            enabled: Whether the stage is enabled
            **kwargs: Additional stage parameters
        """
        stage = PipelineStage(
            name=name,
            pipeline_type=PipelineType.FEATURE,
            pipeline_class=FeaturePipeline,
            config=config,
            depends_on=depends_on or [],
            enabled=enabled,
            **kwargs
        )
        self.add_stage(stage)
    
    def add_training_stage(
        self,
        name: str,
        config: PipelineConfig,
        depends_on: Optional[List[str]] = None,
        enabled: bool = True,
        **kwargs
    ) -> None:
        """
        Add a training pipeline stage to the orchestrator.
        
        Args:
            name: Name of the stage
            config: Pipeline configuration
            depends_on: Optional list of stage names this stage depends on
            enabled: Whether the stage is enabled
            **kwargs: Additional stage parameters
        """
        stage = PipelineStage(
            name=name,
            pipeline_type=PipelineType.TRAINING,
            pipeline_class=TrainingPipeline,
            config=config,
            depends_on=depends_on or [],
            enabled=enabled,
            **kwargs
        )
        self.add_stage(stage)
    
    def add_deployment_stage(
        self,
        name: str,
        config: PipelineConfig,
        depends_on: Optional[List[str]] = None,
        enabled: bool = True,
        **kwargs
    ) -> None:
        """
        Add a deployment pipeline stage to the orchestrator.
        
        Args:
            name: Name of the stage
            config: Pipeline configuration
            depends_on: Optional list of stage names this stage depends on
            enabled: Whether the stage is enabled
            **kwargs: Additional stage parameters
        """
        stage = PipelineStage(
            name=name,
            pipeline_type=PipelineType.DEPLOYMENT,
            pipeline_class=DeploymentPipeline,
            config=config,
            depends_on=depends_on or [],
            enabled=enabled,
            **kwargs
        )
        self.add_stage(stage)
    
    def add_monitoring_stage(
        self,
        name: str,
        config: PipelineConfig,
        depends_on: Optional[List[str]] = None,
        enabled: bool = True,
        **kwargs
    ) -> None:
        """
        Add a monitoring pipeline stage to the orchestrator.
        
        Args:
            name: Name of the stage
            config: Pipeline configuration
            depends_on: Optional list of stage names this stage depends on
            enabled: Whether the stage is enabled
            **kwargs: Additional stage parameters
        """
        stage = PipelineStage(
            name=name,
            pipeline_type=PipelineType.MONITORING,
            pipeline_class=MonitoringPipeline,
            config=config,
            depends_on=depends_on or [],
            enabled=enabled,
            **kwargs
        )
        self.add_stage(stage)
    
    def add_custom_stage(
        self,
        name: str,
        pipeline_class: Type[Pipeline],
        config: PipelineConfig,
        depends_on: Optional[List[str]] = None,
        enabled: bool = True,
        **kwargs
    ) -> None:
        """
        Add a custom pipeline stage to the orchestrator.
        
        Args:
            name: Name of the stage
            pipeline_class: Custom pipeline class
            config: Pipeline configuration
            depends_on: Optional list of stage names this stage depends on
            enabled: Whether the stage is enabled
            **kwargs: Additional stage parameters
        """
        stage = PipelineStage(
            name=name,
            pipeline_type=PipelineType.CUSTOM,
            pipeline_class=pipeline_class,
            config=config,
            depends_on=depends_on or [],
            enabled=enabled,
            **kwargs
        )
        self.add_stage(stage)
    
    def get_execution_order(self) -> List[str]:
        """
        Get the execution order of stages based on dependencies.
        
        Returns:
            List of stage names in execution order
            
        Raises:
            OrchestrationError: If there are cyclic dependencies
        """
        # Only include enabled stages
        enabled_stages = {name: stage for name, stage in self.stages.items() if stage.enabled}
        
        # Track visited and completed stages for cycle detection
        visited: Dict[str, bool] = {name: False for name in enabled_stages}
        temp: Dict[str, bool] = {name: False for name in enabled_stages}
        order: List[str] = []
        
        def visit(node: str) -> None:
            """DFS function to visit nodes and detect cycles."""
            # If node is in temp, we have a cycle
            if temp.get(node, False):
                raise OrchestrationError(f"Cyclic dependency detected involving stage '{node}'")
            
            # If already visited, skip
            if visited.get(node, True):
                return
            
            # Mark as temporary visited (in progress)
            temp[node] = True
            
            # Visit all dependencies
            for dep in enabled_stages.get(node, PipelineStage("", PipelineType.CUSTOM, Pipeline, PipelineConfig())).depends_on:
                if dep not in enabled_stages:
                    self.logger.warning(f"Stage '{node}' depends on disabled or non-existent stage '{dep}'")
                    continue
                visit(dep)
            
            # Mark as visited and add to order
            temp[node] = False
            visited[node] = True
            order.append(node)
        
        # Visit all nodes
        for name in enabled_stages:
            if not visited[name]:
                visit(name)
        
        # Reverse to get the correct order
        return list(reversed(order))
    
    @pipeline_error_handler
    def run_stage(self, stage_name: str) -> Result:
        """
        Run a single pipeline stage.
        
        Args:
            stage_name: Name of the stage to run
            
        Returns:
            Result of the stage execution
            
        Raises:
            OrchestrationError: If the stage does not exist
        """
        if stage_name not in self.stages:
            raise OrchestrationError(f"Stage '{stage_name}' not found")
        
        stage = self.stages[stage_name]
        
        if not stage.enabled:
            self.logger.info(f"Stage '{stage_name}' is disabled, skipping")
            return Result(
                status=StatusEnum.SUCCESS,
                message=f"Stage '{stage_name}' skipped (disabled)",
                details={"stage": stage_name, "skipped": True}
            )
        
        # Check if all dependencies succeeded
        for dep in stage.depends_on:
            if dep not in self.stage_results:
                # Dependency hasn't been run yet
                raise OrchestrationError(f"Dependency '{dep}' of stage '{stage_name}' has not been run")
            
            if self.stage_results[dep].status == StatusEnum.FAILED:
                # Dependency failed
                self.logger.error(f"Dependency '{dep}' of stage '{stage_name}' failed, skipping stage")
                return Result(
                    status=StatusEnum.FAILED,
                    message=f"Stage '{stage_name}' skipped (dependency '{dep}' failed)",
                    details={"stage": stage_name, "failed_dependency": dep}
                )
        
        # Initialize the pipeline
        try:
            pipeline = stage.pipeline_class(stage.config)
        except Exception as e:
            self.logger.exception(f"Failed to initialize pipeline for stage '{stage_name}': {str(e)}")
            return Result(
                status=StatusEnum.FAILED,
                message=f"Failed to initialize pipeline for stage '{stage_name}': {str(e)}",
                errors=[{"type": type(e).__name__, "message": str(e)}]
            )
        
        # Run the pipeline
        self.logger.info(f"Running stage '{stage_name}'")
        result = pipeline.run()
        
        # Handle retries if needed
        retry_count = 0
        while (result.status == StatusEnum.FAILED and 
               stage.on_failure == "retry" and 
               retry_count < stage.max_retries):
            retry_count += 1
            self.logger.warning(f"Retrying stage '{stage_name}' (attempt {retry_count}/{stage.max_retries})")
            result = pipeline.run()
        
        # Update stage retry count
        stage.retry_count = retry_count
        
        # Log the result
        if result.is_success:
            self.logger.info(f"Stage '{stage_name}' completed successfully")
        else:
            self.logger.error(f"Stage '{stage_name}' failed: {result.message}")
        
        return result
    
    def run(self) -> OrchestratorResult:
        """
        Run all pipeline stages in the correct order.
        
        Returns:
            OrchestratorResult: The result of the orchestration
        """
        import time
        start_time = time.time()
        
        # Reset results
        self.stage_results = {}
        failed_stages = []
        skipped_stages = []
        
        try:
            # Get the execution order
            execution_order = self.get_execution_order()
            self.logger.info(f"Execution order: {', '.join(execution_order)}")
            
            # Run each stage in order
            for stage_name in execution_order:
                try:
                    # Run the stage
                    result = self.run_stage(stage_name)
                    self.stage_results[stage_name] = result
                    
                    # Handle failures
                    if result.status == StatusEnum.FAILED:
                        failed_stages.append(stage_name)
                        
                        # Determine what to do on failure
                        on_failure = self.stages[stage_name].on_failure
                        if on_failure == "stop":
                            self.logger.error(f"Stage '{stage_name}' failed with on_failure=stop, aborting orchestration")
                            break
                    
                except Exception as e:
                    # Handle unexpected errors
                    self.logger.exception(f"Unexpected error running stage '{stage_name}': {str(e)}")
                    failed_stages.append(stage_name)
                    self.stage_results[stage_name] = Result(
                        status=StatusEnum.FAILED,
                        message=f"Unexpected error running stage '{stage_name}': {str(e)}",
                        errors=[{"type": type(e).__name__, "message": str(e)}]
                    )
                    break
            
            # Identify skipped stages
            for stage_name in self.stages:
                if stage_name not in self.stage_results and self.stages[stage_name].enabled:
                    skipped_stages.append(stage_name)
            
            # Determine overall status
            status = StatusEnum.SUCCESS if len(failed_stages) == 0 else StatusEnum.FAILED
            
            # Create the result
            execution_time = time.time() - start_time
            result = OrchestratorResult(
                status=status,
                message=self._create_result_message(status, execution_order, failed_stages, skipped_stages),
                stages=[stage.to_dict() for stage in self.stages.values()],
                stage_results=self.stage_results,
                failed_stages=failed_stages,
                skipped_stages=skipped_stages,
                execution_time_seconds=execution_time,
                details={
                    "execution_order": execution_order,
                    "total_stages": len(self.stages),
                    "completed_stages": len(self.stage_results),
                    "failed_stages": len(failed_stages),
                    "skipped_stages": len(skipped_stages),
                    "execution_time_seconds": execution_time
                }
            )
            
            self.logger.info(f"Orchestration completed in {execution_time:.2f} seconds with status: {status}")
            return result
            
        except Exception as e:
            # Handle orchestration errors
            execution_time = time.time() - start_time
            self.logger.exception(f"Orchestration error: {str(e)}")
            
            return OrchestratorResult(
                status=StatusEnum.FAILED,
                message=f"Orchestration failed: {str(e)}",
                stages=[stage.to_dict() for stage in self.stages.values()],
                stage_results=self.stage_results,
                failed_stages=failed_stages,
                skipped_stages=skipped_stages,
                execution_time_seconds=execution_time,
                errors=[{"type": type(e).__name__, "message": str(e)}]
            )
    
    def _create_result_message(
        self, status: StatusEnum, execution_order: List[str], failed_stages: List[str], skipped_stages: List[str]
    ) -> str:
        """
        Create a human-readable message for the orchestration result.
        
        Args:
            status: Overall status
            execution_order: Execution order of stages
            failed_stages: List of failed stage names
            skipped_stages: List of skipped stage names
            
        Returns:
            Human-readable message
        """
        if status == StatusEnum.SUCCESS:
            return f"All {len(execution_order)} stages completed successfully"
        else:
            completed = len(execution_order) - len(failed_stages) - len(skipped_stages)
            return (
                f"Orchestration completed with {len(failed_stages)} failed stages, "
                f"{completed} successful stages, and {len(skipped_stages)} skipped stages"
            )
