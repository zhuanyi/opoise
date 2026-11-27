"""Type registries: categories and the classes registered under them.

A registry stores constraint or objective models by their type tag, so a
document can name a class by tag and any code can look the class up again. A
category names a group of tags, and a class declares the category it belongs
to. Categories and types are process-global and filled at import time; a
project registers its own category and classes at import time too. Registration
is single-threaded and takes no lock.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from .errors import OpoiseError

_CATEGORY_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
_TYPE_TAG = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_KINDS = ("constraint", "objective")
_CATEGORIES: dict[str, dict[str, Category]] = {kind: {} for kind in _KINDS}
_REGISTRIES: dict[str, list[Registry]] = {kind: [] for kind in _KINDS}


@dataclass(frozen=True, slots=True)
class Category:
    """A named group of constraint or objective types.

    Attributes
    ----------
    name : str
        Uppercase name, at most 32 characters from A-Z, 0-9 and underscore,
        starting with a letter, e.g. "RISK".
    description : str
        One generic sentence on what the category covers.
    kind : str
        "constraint" or "objective". The two kinds have separate registries,
        so one name can exist once per kind.
    """

    name: str
    description: str
    kind: Literal["constraint", "objective"]


def register_category(category: Category) -> Category:
    """Register a category and return it unchanged.

    Registering the same category twice is allowed while the description
    matches, so a module can be imported again without failing.

    Parameters
    ----------
    category : Category
        The category to make available to classes of its kind.

    Returns
    -------
    Category
        The category passed in.

    Raises
    ------
    OpoiseError
        If the kind is unknown, the name does not match its pattern, the
        description is empty, or the name is registered with another
        description ("category 'X' is already registered").
    """
    if category.kind not in _CATEGORIES:
        raise OpoiseError(f"category {category.name!r}: unknown kind {category.kind!r}")
    if not _CATEGORY_NAME.match(category.name):
        raise OpoiseError(f"category {category.name!r} must match {_CATEGORY_NAME.pattern}")
    if not category.description.strip():
        raise OpoiseError(f"category {category.name!r} needs a description")
    table = _CATEGORIES[category.kind]
    existing = table.get(category.name)
    if existing is not None and existing.description != category.description:
        raise OpoiseError(f"category {category.name!r} is already registered")
    table[category.name] = category
    return category


def categories(kind: str | None = None) -> list[Category]:
    """Return the registered categories, sorted by name and kind.

    Parameters
    ----------
    kind : str or None
        "constraint", "objective" or None for both kinds.

    Returns
    -------
    list of Category
        A new list in sorted order; mutating it does not change the registry.

    Raises
    ------
    OpoiseError
        If kind is neither "constraint", "objective" nor None.
    """
    if kind is not None and kind not in _CATEGORIES:
        raise OpoiseError(f"unknown kind {kind!r}")
    kinds = _KINDS if kind is None else (kind,)
    found = [category for one in kinds for category in _CATEGORIES[one].values()]
    return sorted(found, key=lambda category: (category.name, category.kind))


def unregister_category(name: str, kind: str) -> None:
    """Remove a category of one kind.

    Parameters
    ----------
    name : str
        Category name.
    kind : str
        "constraint" or "objective".

    Raises
    ------
    OpoiseError
        If the category is not registered, or classes are still registered
        under it ("category 'X' has registered types: a, b; unregister them
        first").
    """
    table = _CATEGORIES.get(kind)
    if table is None or name not in table:
        raise OpoiseError(f"category {name!r} is not registered")
    users = sorted(tag for registry in _REGISTRIES[kind] for tag in registry.types(name))
    if users:
        raise OpoiseError(
            f"category {name!r} has registered types: {', '.join(users)}; unregister them first"
        )
    del table[name]


class Registry:
    """The classes of one kind, keyed by their type tag.

    Attributes
    ----------
    base : type
        Class every registered type must subclass.
    kind : str
        "constraint" or "objective"; also selects the categories a type may
        belong to.
    """

    def __init__(self, base: type, kind: str) -> None:
        """Create a registry and make it visible to unregister_category.

        Parameters
        ----------
        base : type
            Class every registered type must subclass.
        kind : str
            "constraint" or "objective".

        Raises
        ------
        OpoiseError
            If kind is neither "constraint" nor "objective".
        """
        if kind not in _CATEGORIES:
            raise OpoiseError(f"unknown kind {kind!r}")
        self.base = base
        self.kind = kind
        self._types: dict[str, type] = {}
        _REGISTRIES[kind].append(self)

    def register_category(self, category: Category) -> Category:
        """Register a category of this registry's kind and return it.

        Parameters
        ----------
        category : Category
            The category to register.

        Returns
        -------
        Category
            The category passed in.

        Raises
        ------
        OpoiseError
            If the category belongs to the other kind, or for any reason
            register_category rejects it.
        """
        if category.kind != self.kind:
            raise OpoiseError(f"category {category.name!r} is a {category.kind} category")
        return register_category(category)

    def categories(self) -> list[Category]:
        """Return the categories of this registry's kind, sorted by name.

        Returns
        -------
        list of Category
            A new list in sorted order.
        """
        return categories(self.kind)

    def unregister_category(self, name: str) -> None:
        """Remove a category of this registry's kind.

        Parameters
        ----------
        name : str
            Category name.

        Raises
        ------
        OpoiseError
            If the category is not registered or still has registered types.
        """
        unregister_category(name, self.kind)

    def register(self, cls: type, replace: bool = False) -> type:
        """Register a class under its type tag and return it.

        The class must subclass the registry base, declare its type tag in its
        own class body, belong to a registered category of the same kind, and
        override build().

        Parameters
        ----------
        cls : type
            The class to register.
        replace : bool
            Replace a class already registered under the same tag.

        Returns
        -------
        type
            The class passed in, so register can be used as a decorator.

        Raises
        ------
        OpoiseError
            If the class fails a check, or its tag is taken and replace is
            False ("constraint type 'x' is already registered").
        """
        if not (isinstance(cls, type) and issubclass(cls, self.base)):
            name = getattr(cls, "__name__", repr(cls))
            raise OpoiseError(f"'{name}' must subclass {self.base.__name__}")
        declared = inspect.get_annotations(cls, eval_str=False)
        field = cls.model_fields.get("type")
        if "type" not in declared or field is None:
            raise OpoiseError(f"'{cls.__name__}': declare type: Literal[...] in the class body")
        values = get_args(field.annotation) if get_origin(field.annotation) is Literal else ()
        tag = field.default
        if len(values) != 1 or values[0] != tag or not isinstance(tag, str) or not _TYPE_TAG.match(tag):
            raise OpoiseError(
                f"'{cls.__name__}': the type tag must be one lowercase Literal equal to the default, "
                "e.g. Literal['my_type'] = 'my_type'"
            )
        category = getattr(cls, "category", None)
        if not isinstance(category, Category):
            raise OpoiseError(f"'{cls.__name__}': set category to a registered Category")
        if _CATEGORIES[self.kind].get(category.name) is None:
            other = "objective" if self.kind == "constraint" else "constraint"
            if category.name in _CATEGORIES[other]:
                article = "an" if other == "objective" else "a"
                raise OpoiseError(
                    f"'{cls.__name__}': category {category.name!r} is {article} {other} category"
                )
            raise OpoiseError(f"'{cls.__name__}': category {category.name!r} is not registered")
        if cls.build is self.base.build:
            raise OpoiseError(f"'{cls.__name__}' must override build()")
        if tag in self._types and not replace:
            raise OpoiseError(f"{self.kind} type '{tag}' is already registered")
        self._types[tag] = cls
        return cls

    def unregister(self, type_tag: str, missing_ok: bool = False) -> None:
        """Remove a type by tag.

        Parameters
        ----------
        type_tag : str
            The type tag to remove.
        missing_ok : bool
            Return quietly instead of raising when the tag is unknown.

        Raises
        ------
        OpoiseError
            If the tag is unknown and missing_ok is False ("unknown
            constraint type 'x'").
        """
        if type_tag not in self._types:
            if missing_ok:
                return
            raise OpoiseError(f"unknown {self.kind} type '{type_tag}'")
        del self._types[type_tag]

    def types(self, category: str | Category | None = None) -> dict[str, type]:
        """Return the registered types, keyed by tag and sorted by tag.

        Parameters
        ----------
        category : str or Category or None
            Keep only the types of this category (by name); None keeps all.

        Returns
        -------
        dict of str to type
            A new dict in sorted order; mutating it does not change the
            registry.
        """
        if category is None:
            return dict(sorted(self._types.items()))
        name = category.name if isinstance(category, Category) else category
        return dict(sorted((tag, cls) for tag, cls in self._types.items() if cls.category.name == name))

    def get(self, type_tag: str) -> type:
        """Return the class registered under one tag.

        Parameters
        ----------
        type_tag : str
            The type tag.

        Returns
        -------
        type
            The registered class.

        Raises
        ------
        ValueError
            If the tag is not registered ("unknown constraint type 'x'").
        """
        try:
            return self._types[type_tag]
        except KeyError:
            raise ValueError(f"unknown {self.kind} type '{type_tag}'") from None


def dispatch(value: Any, registry: Registry) -> Any:
    """Turn a list of tagged dicts into the registered instances.

    Instances of a registered subclass pass through unchanged, so a validated
    model can be rebuilt from its own dump. A bare base instance and a dict
    without a tag are rejected here so neither reaches a solver. Anything else
    is left for pydantic to report.

    Parameters
    ----------
    value : Any
        The value pydantic received for a list field.
    registry : Registry
        The registry the dicts are resolved against.

    Returns
    -------
    Any
        A new list with one entry per input entry, or the input unchanged when
        it is not a list or a tuple.

    Raises
    ------
    ValueError
        If an entry is a dict without a "type" tag ("constraint entry 0 has no
        'type' tag; registered: ..."), if an entry is a bare base instance
        ("constraint entry 1 is a bare ConstraintBase; use a registered
        constraint type"), or if a dict names a tag that is not registered
        ("unknown constraint type 'x'; registered: a, b"). Pydantic reports
        each as a ValidationError.
    """
    if not isinstance(value, (list, tuple)):
        return value
    known = registry.types()
    resolved: list[Any] = []
    for i, item in enumerate(value):
        if isinstance(item, registry.base):
            if type(item) is registry.base:
                raise ValueError(
                    f"{registry.kind} entry {i} is a bare {registry.base.__name__}; "
                    f"use a registered {registry.kind} type"
                )
            resolved.append(item)
        elif isinstance(item, dict):
            if "type" not in item:
                raise ValueError(
                    f"{registry.kind} entry {i} has no 'type' tag; "
                    f"registered: {', '.join(sorted(known))}"
                )
            tag = item["type"]
            if tag not in known:
                raise ValueError(
                    f"unknown {registry.kind} type {tag!r}; registered: {', '.join(sorted(known))}"
                )
            resolved.append(known[tag](**item))
        else:
            resolved.append(item)
    return resolved
