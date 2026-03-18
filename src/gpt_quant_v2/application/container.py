"""Dependency injection container for managing application dependencies."""

from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

T = TypeVar("T")


class Lifetime(Enum):
    """Defines the lifetime of a registered dependency."""

    SINGLETON = "singleton"
    TRANSIENT = "transient"


class Container:
    """Simple dependency injection container.

    Supports registering and resolving dependencies with different lifetimes:
    - SINGLETON: Same instance is returned on every resolve
    - TRANSIENT: New instance is created on every resolve

    Example:
        container = Container()

        # Register a singleton service
        container.register(DatabaseService, lifetime=Lifetime.SINGLETON)

        # Register with a factory function
        container.register(Logger, factory=lambda c: Logger(c.resolve(Config)))

        # Register an existing instance
        container.register_instance(Config, config_instance)

        # Resolve dependencies
        db = container.resolve(DatabaseService)
    """

    def __init__(self) -> None:
        """Initialize an empty container."""
        self._registrations: Dict[Type, Dict[str, Any]] = {}
        self._singletons: Dict[Type, Any] = {}

    def register(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[["Container"], T]] = None,
        lifetime: Lifetime = Lifetime.TRANSIENT,
    ) -> "Container":
        """Register a service type with its implementation or factory.

        Args:
            service_type: The type to register (interface or base class).
            implementation: The concrete implementation type (optional).
            factory: A factory function that creates instances (optional).
            lifetime: The lifetime of the service (default: TRANSIENT).

        Returns:
            The container instance for method chaining.

        Raises:
            ValueError: If neither implementation nor factory is provided,
                       or if both are provided.
        """
        if implementation is None and factory is None:
            # Use service_type as its own implementation
            implementation = service_type
        elif implementation is not None and factory is not None:
            raise ValueError(
                "Cannot specify both implementation and factory. Choose one."
            )

        self._registrations[service_type] = {
            "implementation": implementation,
            "factory": factory,
            "lifetime": lifetime,
        }

        return self

    def register_instance(
        self,
        service_type: Type[T],
        instance: T,
    ) -> "Container":
        """Register an existing instance as a singleton.

        Args:
            service_type: The type to register.
            instance: The pre-created instance to use.

        Returns:
            The container instance for method chaining.
        """
        self._registrations[service_type] = {
            "implementation": None,
            "factory": None,
            "lifetime": Lifetime.SINGLETON,
        }
        self._singletons[service_type] = instance

        return self

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service type to its implementation instance.

        Args:
            service_type: The type to resolve.

        Returns:
            An instance of the requested type.

        Raises:
            KeyError: If the service type is not registered.
        """
        if service_type not in self._registrations:
            raise KeyError(f"Service type {service_type.__name__} is not registered")

        registration = self._registrations[service_type]
        lifetime = registration["lifetime"]

        # Return existing singleton if available
        if lifetime == Lifetime.SINGLETON and service_type in self._singletons:
            return self._singletons[service_type]

        # Create new instance
        instance = self._create_instance(registration)

        # Cache singleton
        if lifetime == Lifetime.SINGLETON:
            self._singletons[service_type] = instance

        return instance

    def _create_instance(self, registration: Dict[str, Any]) -> Any:
        """Create an instance based on registration configuration."""
        factory = registration["factory"]
        implementation = registration["implementation"]

        if factory is not None:
            return factory(self)

        if implementation is not None:
            return implementation()

        raise ValueError("No implementation or factory available")

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered.

        Args:
            service_type: The type to check.

        Returns:
            True if the type is registered, False otherwise.
        """
        return service_type in self._registrations

    def clear(self) -> None:
        """Clear all registrations and cached singletons."""
        self._registrations.clear()
        self._singletons.clear()

    def create_scope(self) -> "Container":
        """Create a child container that inherits parent registrations.

        The child container can override parent registrations without
        affecting the parent. Useful for request-scoped dependencies.

        Returns:
            A new container with inherited registrations.
        """
        scoped = Container()
        scoped._registrations = dict(self._registrations)
        # Singletons are shared with parent
        scoped._singletons = self._singletons
        return scoped
