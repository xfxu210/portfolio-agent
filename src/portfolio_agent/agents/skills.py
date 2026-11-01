from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AgentSkill:
    """A versioned, load-on-demand investment-research playbook."""

    name: str
    version: str
    description: str
    when_to_use: str
    target_scope: str
    tools: tuple[str, ...]
    instructions: str
    source: Path

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "target_scope": self.target_scope,
            "tools": self.tools,
        }


class SkillRegistry:
    """Loads auditable skill instructions from the package skill library."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).with_name("skill_library")
        self._skills = self._load()

    def _load(self) -> dict[str, AgentSkill]:
        skills: dict[str, AgentSkill] = {}
        for path in sorted(self.root.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            if not text.startswith("---\n"):
                raise ValueError(f"skill is missing YAML frontmatter: {path}")
            try:
                header, body = text[4:].split("\n---\n", 1)
            except ValueError as exc:
                raise ValueError(f"skill frontmatter is not terminated: {path}") from exc
            metadata = yaml.safe_load(header)
            if not isinstance(metadata, dict):
                raise ValueError(f"skill frontmatter must be a mapping: {path}")
            required = {
                "name",
                "version",
                "description",
                "when_to_use",
                "target_scope",
                "tools",
            }
            missing = required - set(metadata)
            if missing:
                raise ValueError(f"skill {path.name} is missing {sorted(missing)}")
            name = str(metadata["name"])
            if name in skills:
                raise ValueError(f"duplicate skill name: {name}")
            scope = str(metadata["target_scope"])
            if scope not in {"global", "company", "portfolio", "cross_section"}:
                raise ValueError(f"skill {name} has invalid target_scope {scope}")
            tools = tuple(str(value) for value in metadata["tools"])
            skills[name] = AgentSkill(
                name=name,
                version=str(metadata["version"]),
                description=str(metadata["description"]),
                when_to_use=str(metadata["when_to_use"]),
                target_scope=scope,
                tools=tools,
                instructions=body.strip(),
                source=path,
            )
        if not skills:
            raise ValueError(f"no skills found in {self.root}")
        return skills

    def get(self, name: str) -> AgentSkill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise ValueError(f"unknown skill: {name}") from exc

    def catalog(self) -> list[dict[str, Any]]:
        return [self._skills[name].catalog_entry() for name in sorted(self._skills)]

    def validate_tools(self, available_tools: set[str]) -> None:
        for skill in self._skills.values():
            unknown = set(skill.tools) - available_tools
            if unknown:
                raise ValueError(f"skill {skill.name} references unknown tools: {sorted(unknown)}")

