from moe.base.expert import BaseExpert
from moe.base.mixture import BaseMoE
from moe.default.strategies import DefaultExpertStrategy, DefaultMoEStrategy
from moe.prebuilt.router.moe import AutonomousRouter
from langchain_core.runnables import RunnableLambda



def Expert(strategy=DefaultExpertStrategy):
    """Decorator to convert a class into an expert with the given strategy."""
    def wrapper(cls):
        class WrappedExpert(BaseExpert):
            def __init__(self, *args, **kwargs):
                name = kwargs.get('name', None) or cls.__name__
                description = kwargs.get('description', None) or cls.__doc__
                agent = kwargs.get('agent', None) or cls.agent

                # Delete static agent
                if hasattr(cls, 'agent'):
                    delattr(cls, 'agent')

                super().__init__(agent=agent, description=description, name=name, strategy=strategy())

        return WrappedExpert
    return wrapper


from functools import wraps


# ForceFirst decorator to force first expert
def ForceFirst(next_expert_name):
    def decorator(cls):
        cls.router = BaseExpert(
            name='Router',
            description='Static router',
            strategy=DefaultExpertStrategy(),
            agent=RunnableLambda(lambda state: (
                f"\nAction: {next_expert_name}"
                f"\nAction Input: {state['input']}"
            ))
        )
        return cls
    return decorator

# Autonomous decorator to store the router instance
def Autonomous(llm):
    def decorator(cls):
        cls.router = AutonomousRouter(llm)  # Store router on the class
        return cls
    return decorator

# MoE decorator to wrap the class and initialize it properly
def MoE(strategy=DefaultMoEStrategy):
    def wrapper(cls):
        class WrappedMoE(BaseMoE):
            def __init__(self, *args, **kwargs):
                name = kwargs.get('name', None) or cls.__name__
                description = kwargs.get('description', None) or cls.__doc__
                experts = kwargs.get('experts') or cls.experts
                router = kwargs.get('router', None) or getattr(cls, 'router', None)
                verbose = kwargs.get('verbose', None) or 0

                # Delete static experts
                if hasattr(cls, 'experts'):
                    delattr(cls, 'experts')
                    
                # Delete static router
                if hasattr(cls, 'router'):
                    delattr(cls, 'router')

                super().__init__(
                    name=name,
                    description=description,
                    strategy=strategy(),
                    verbose=verbose,
                    experts=experts,
                    router=router,
                )

        return WrappedMoE
    return wrapper



