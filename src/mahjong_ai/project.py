"""Project manifests and layout paths shared by research and competition tools."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ProjectDescriptor:
    """The identity and stable storage roots of one strategy project.

    Every active project has an explicit manifest. The descriptor owns the
    project-local `.local` layout used by research and training.
    """

    root: Path
    id: str
    family: str
    entrypoint: str | None = None
    dependencies: tuple[str, ...] = ()

    @classmethod
    def load(cls, root: str | Path) -> "ProjectDescriptor":
        path = Path(root).resolve()
        manifest = path / "project.toml"
        if not manifest.exists():
            raise ValueError(f"project manifest not found: {manifest}")
        with manifest.open("rb") as stream:
            data = tomllib.load(stream)
        project = data.get("project", data)
        project_id = project.get("id", path.name)
        family = project.get("family", "custom")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project.toml requires a non-empty project.id")
        if not isinstance(family, str) or not family:
            raise ValueError("project.toml requires a non-empty project.family")
        deps = project.get("dependencies", data.get("dependencies", []))
        if not isinstance(deps, list) or not all(
            isinstance(item, str) for item in deps
        ):
            raise ValueError("project dependencies must be a list of strings")
        entrypoint = project.get("entrypoint")
        if entrypoint is not None and not isinstance(entrypoint, str):
            raise ValueError("project entrypoint must be a string")
        return cls(path, project_id, family, entrypoint, tuple(deps))

    @property
    def local(self) -> Path:
        return self.root / ".local"

    @property
    def research(self) -> Path:
        return self.local / "research"

    @property
    def strategies(self) -> Path:
        return self.local / "strategies"

    @property
    def service(self) -> Path:
        return self.local / "service"

    @property
    def releases(self) -> Path:
        return self.root / "releases"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def workspace(self) -> Path:
        """Repository workspace used by a project, when explicitly declared."""
        manifest = self.root / "project.toml"
        if not manifest.exists():
            return self.root
        with manifest.open("rb") as stream:
            data = tomllib.load(stream)
        value = data.get("project", data).get("workspace")
        return (self.root / value).resolve() if value else self.root
