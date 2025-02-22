from moe.base.expert import BaseExpert
from moe.base.mixture import BaseMoE
from moe.default.strategies import DefaultExpertStrategy, DefaultMoEStrategy
from moe.prebuilt.router.moe import AutonomousRouter
from langchain_core.runnables import RunnableLambda



def Expert(strategy=DefaultExpertStrategy):
    """Decorator to convert a class into an expert with the given strategy."""
    if not callable(strategy):
        raise TypeError("The provided strategy must be callable.")

    def wrapper(cls):
        class WrappedExpert(BaseExpert):
            def __init__(self, *args, **kwargs):
                try:
                    name = kwargs.get('name') or cls.__name__
                    description = kwargs.get('description') or cls.__doc__
                    agent = kwargs.get('agent') or getattr(cls, 'agent', None)

                    if agent is None:
                        raise RuntimeError(f"agent cannot be None. Provide your agent as a static element of {cls.__name__}")
                    
                    if description is None or not len(description):
                        raise RuntimeError(f"description cannot be empty. Provide docstring for your expert class")

                    # Delete static agent safely
                    if hasattr(cls, 'agent'):
                        try:
                            delattr(cls, 'agent')
                        except AttributeError as e:
                            raise RuntimeError(f"Failed to remove 'agent' attribute from {cls.__name__}: {e}")

                    super().__init__(agent=agent, description=description, name=name, strategy=strategy())
                except Exception as e:
                    raise RuntimeError(f"Error initializing WrappedExpert for {cls.__name__}: {e}")

        return WrappedExpert
    return wrapper


# ForceFirst decorator to force first expert
def ForceFirst(next_expert_name):
    if not isinstance(next_expert_name, str) or not next_expert_name.strip():
        raise ValueError("Error at @ForceFirst: `next_expert_name` must be a non-empty string.")

    def decorator(cls):
        try:
            cls.router = BaseExpert(
                name='Router',
                description='Static router',
                strategy=DefaultExpertStrategy(),
                agent=RunnableLambda(lambda state: (
                    f"\nAction: {next_expert_name}"
                    f"\nAction Input: {state.get('input', 'No Input')}"
                ))
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing router for {cls.__name__}: {e}")

        return cls
    return decorator

# Autonomous decorator to store the router instance
def Autonomous(llm):
    if llm is None:
        raise ValueError("error at @Autonomous: llm cannot be None.")

    def decorator(cls):
        try:
            cls.router = AutonomousRouter(llm)  # Store router on the class
        except Exception as e:
            raise RuntimeError(f"Error initializing AutonomousRouter for {cls.__name__}: {e}")

        return cls
    return decorator

# MoE decorator to wrap the class and initialize it properly
def MoE(strategy=DefaultMoEStrategy):
    """Decorator for mixture of experts initialization."""
    if not callable(strategy):
        raise TypeError("The provided strategy must be callable.")

    def wrapper(cls):
        class WrappedMoE(BaseMoE):
            def __init__(self, *args, **kwargs):
                try:
                    name = kwargs.get('name') or cls.__name__
                    description = kwargs.get('description') or cls.__doc__
                    experts = kwargs.get('experts') or getattr(cls, 'experts', None)
                    router = kwargs.get('router') or getattr(cls, 'router', None)
                    verbose = kwargs.get('verbose', 0)

                    if router is None:
                        raise RuntimeError(f"router cannot be None. Anotate your moe with @Autonomous or @ForceFirst")
                    
                    if description is None or not len(description):
                        raise RuntimeError(f"description cannot be empty. Provide docstring for your expert class")
                    
                    if experts is None:
                        raise RuntimeError(f"You must declare your `experts` array as a static element of the class {cls.__name__}")

                    if not isinstance(verbose, int) or verbose < 0:
                        raise ValueError("verbose must be a non-negative integer.")

                    # Delete static experts safely
                    if hasattr(cls, 'experts'):
                        try:
                            delattr(cls, 'experts')
                        except AttributeError as e:
                            raise RuntimeError(f"Failed to remove 'experts' attribute from {cls.__name__}: {e}")
                    
                    # Delete static router safely
                    if hasattr(cls, 'router'):
                        try:
                            delattr(cls, 'router')
                        except AttributeError as e:
                            raise RuntimeError(f"Failed to remove 'router' attribute from {cls.__name__}: {e}")

                    super().__init__(
                        name=name,
                        description=description,
                        strategy=strategy(),
                        verbose=verbose,
                        experts=experts,
                        router=router,
                    )
                except Exception as e:
                    raise RuntimeError(f"Error initializing WrappedMoE for {cls.__name__}: {e}")

        return WrappedMoE
    return wrapper
