from moe.base.expert import BaseExpert
from moe.base.mixture import BaseMoE
from moe.base.strategies import BaseExpertStrategy, BaseMoEStrategy
from moe.default.strategies import DefaultExpertStrategy, DefaultMoEStrategy
from moe.prebuilt.router.moe import AutonomousRouter
from langchain_core.runnables import RunnableLambda

############################## Decorators for Experts ##########################################

# ForceNext decorator to force next expert
def ForceNext(expert_name):
    if not isinstance(expert_name, str) or not expert_name.strip():
        raise ValueError("Error at @ForceNext: `expert_name` must be a non-empty string.")

    def decorator(cls):
        try:
            # if not issubclass(cls, BaseExpert):
            #     raise TypeError(f"@ForceNext must be used only with a derivative of BaseExpert, but tried to use it with {cls.__name__}")
            if hasattr(cls, 'decorators'):
                cls.decorators.add('@ForceNext')
            else:
                cls.decorators = {'@ForceNext'}
            
            cls.next = expert_name

        except Exception as e:
            raise RuntimeError(f"Error initializing router for {cls.__name__}: {e}")

        return cls
    return decorator

def Expert(strategy=DefaultExpertStrategy):
    """Decorator to convert a class into an expert with the given strategy."""
    if not callable(strategy):
        raise TypeError('strategy must be callable')

    def wrapper(cls):
        _supported_decorators = {'@Expert', '@ForceNext'}
        if hasattr(cls, 'decorators'):
            cls.decorators.add('@Expert')
        else:
            cls.decorators = {'@Expert'}
        if len(cls.decorators - _supported_decorators) > 0:
            raise TypeError(f"Cannot use the following decorator(s) with an BaseExpert: {cls.decorators - _supported_decorators}. The supported decorators are : {_supported_decorators}")
        
        class WrappedExpert(BaseExpert):
            def __init__(self, *args, **kwargs):
                try:
                    _name = kwargs.get('name') or cls.__name__
                    _description = kwargs.get('description') or cls.__doc__
                    _agent = kwargs.get('agent') or getattr(cls, 'agent', None)
                    _strategy = None

                    if _agent is None:
                        raise RuntimeError(f"agent cannot be None. Provide your agent as a static element of {cls.__name__}")
                    
                    if _description is None or not len(_description):
                        raise RuntimeError(f"description cannot be empty. Provide docstring for your expert class")

                    if isinstance(strategy, BaseExpertStrategy):
                        class _Strategy(BaseExpertStrategy):
                            def execute(self, expert, state):
                                state = strategy.execute(expert, state)
                                state['next'] = self.next or state.get('next')
                                return state

                        _strategy = _Strategy(getattr(cls, 'next', strategy.next))

                    elif issubclass(strategy, BaseExpertStrategy):
                        class _Strategy(BaseExpertStrategy):
                            def execute(self, expert, state):
                                state = strategy().execute(expert, state)
                                state['next'] = self.next or state.get('next')
                                return state

                        _strategy = _Strategy(getattr(cls, 'next', None))
                    
                    else:
                        raise TypeError(f"Strategy must be a subclass or instance of [BaseExpertStrategy, BaseMoEStrategy]. Got {type(strategy)}")

                    # Delete static agent safely
                    if hasattr(cls, 'agent'):
                        try:
                            delattr(cls, 'agent')
                        except AttributeError as e:
                            raise RuntimeError(f"Failed to remove 'agent' attribute from {cls.__name__}: {e}")

                    super().__init__(agent=_agent, description=_description, name=_name, strategy=_strategy)
                except Exception as e:
                    raise RuntimeError(f"Error initializing WrappedExpert for {cls.__name__}: {e}")

        return WrappedExpert
    return wrapper

############################## Decorators for MoEs ##########################################

# Deterministic decorator to force first expert
def Deterministic(expert_name):
    if not isinstance(expert_name, str) or not expert_name.strip():
        raise ValueError("Error at @Deterministic: `expert_name` must be a non-empty string.")

    def decorator(cls):
        try:
            # if not issubclass(cls, BaseMoE):
            #     raise TypeError(f"@Deterministic must be used only with a derivative of BaseMoE, but tried to use it with {cls.__name__}")
            if hasattr(cls, 'decorators'):
                cls.decorators.add('@Deterministic')
            else:
                cls.decorators = {'@Deterministic'}

            cls.router = BaseExpert(
                name='Router',
                description='Determinitstic router',
                strategy=DefaultExpertStrategy(),
                agent=RunnableLambda(lambda state: (
                    f"\nAction: {expert_name}"
                    f"\nAction Input: {state['input']}"
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
            # if not issubclass(cls, BaseMoE):
            #     raise TypeError(f"@Autonomous must be used only with a derivative of BaseMoE, but tried to use it with {cls.__name__}")
            if hasattr(cls, 'decorators'):
                cls.decorators.add('@Autonomous')
            else:
                cls.decorators = {'@Autonomous'}

            cls.router = AutonomousRouter(llm)  # Store router on the class
        except Exception as e:
            raise RuntimeError(f"Error initializing AutonomousRouter for {cls.__name__}: {e}")

        return cls
    return decorator

# MoE decorator to wrap the class and initialize it properly
def MoE(strategy=DefaultMoEStrategy):
    """Decorator for mixture of experts initialization."""
    if not callable(strategy):
        raise TypeError('strategy must be callable')
    
    def wrapper(cls):
        _supported_decorators = {'@MoE', '@Deterministic', '@Autonomous'}
        if hasattr(cls, 'decorators'):
            cls.decorators.add('@MoE')
        else:
            cls.decorators = {'@MoE'}
        
        if len(cls.decorators - _supported_decorators) > 0:
            raise TypeError(f"Cannot use the following decorator(s) with a BaseMoE: {cls.decorators - _supported_decorators}. The supported decorators are : {_supported_decorators}")

        class WrappedMoE(BaseMoE):
            def __init__(self, *args, **kwargs):
                try:
                    _name = kwargs.get('name') or cls.__name__
                    _description = kwargs.get('description') or cls.__doc__
                    _experts = kwargs.get('experts') or getattr(cls, 'experts', None)
                    _router = kwargs.get('router') or getattr(cls, 'router', None)
                    _verbose = kwargs.get('verbose', 0)
                    _strategy = None

                    if _router is None:
                        raise RuntimeError(f"router cannot be None. Anotate your moe with @Autonomous or @ForceFirst")
                    
                    if _description is None or not len(_description):
                        raise RuntimeError(f"description cannot be empty. Provide docstring for your expert class")
                    
                    if _experts is None:
                        raise RuntimeError(f"You must declare your `experts` array as a static element of the class {cls.__name__}")

                    if not isinstance(_verbose, int) or _verbose < 0:
                        raise ValueError("verbose must be a non-negative integer.")
                    
                    if isinstance(strategy, BaseMoEStrategy):
                        _strategy = strategy

                    elif issubclass(strategy, BaseMoEStrategy):
                        _strategy = strategy()
                    
                    else:
                       raise ValueError(f"strategy must be a derivative of `BaseMoEStrategy`, but got `{type(strategy.__class__)}`")

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
                        name=_name,
                        description=_description,
                        strategy=_strategy,
                        verbose=_verbose,
                        experts=_experts,
                        router=_router,
                    )
                except Exception as e:
                    raise RuntimeError(f"Error initializing WrappedMoE for {cls.__name__}: {e}")

        return WrappedMoE
    return wrapper
