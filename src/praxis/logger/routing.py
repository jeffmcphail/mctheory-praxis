"""
Routing matrix.

§18.6: Defines which adapters receive records at which levels/tags.
The "one logger, many destinations" mechanism.

Routing rules:
1. Global min_level on each adapter (adapter.min_level)
2. critical_override: CRITICAL goes everywhere regardless
3. tag_routes: specific tags always routed to specific adapters
4. Default: record goes to adapter if record.level >= adapter.min_level
"""

from dataclasses import dataclass, field

from praxis.logger.records import LogRecord, LogLevel


@dataclass
class RoutingRule:
    """A tag-specific routing override."""
    tag: str
    adapter_names: list[str]
    min_level: int = 0  # Route this tag at any level (0 = always)


class RoutingMatrix:
    """
    §18.6: Routes log records to appropriate adapters.

    Decision logic for each adapter:
    1. CRITICAL override → always route
    2. Tag route exists for any of the record's tags → route to listed adapters
    3. record.level >= adapter.min_level → route
    """

    def __init__(self, critical_override: bool = True):
        self.critical_override = critical_override
        self._tag_routes: dict[str, RoutingRule] = {}  # tag → rule

    def add_tag_route(self, tag: str, adapter_names: list[str], min_level: int = 0) -> None:
        """
        §18.6 tag_routes: Specific tags always routed to specific adapters.
        E.g., trade_cycle → [terminal, database, agent]
        """
        self._tag_routes[tag] = RoutingRule(
            tag=tag, adapter_names=adapter_names, min_level=min_level
        )

    def remove_tag_route(self, tag: str) -> bool:
        """Remove a tag-specific routing rule. Returns True if existed."""
        return self._tag_routes.pop(tag, None) is not None

    def should_route(self, record: LogRecord, adapter_name: str, adapter_min_level: int) -> bool:
        """
        Determine if a record should be routed to a specific adapter.

        Returns True if:
        - CRITICAL override is active and record is CRITICAL
        - Any tag in the record has a tag_route that includes this adapter
        - record.level >= adapter's min_level
        """
        # 1. Critical override
        if self.critical_override and record.level >= LogLevel.CRITICAL:
            return True

        # 2. Tag-specific routing
        for tag in record.tags:
            if tag in self._tag_routes:
                rule = self._tag_routes[tag]
                if adapter_name in rule.adapter_names:
                    if record.level >= rule.min_level:
                        return True

        # 3. Default: adapter min_level filter
        return record.level >= adapter_min_level

    def get_target_adapters(
        self, record: LogRecord, adapters: dict[str, int]
    ) -> set[str]:
        """
        Given a record and a dict of {adapter_name: min_level},
        return the set of adapter names that should receive this record.
        """
        targets = set()
        for adapter_name, adapter_min_level in adapters.items():
            if self.should_route(record, adapter_name, adapter_min_level):
                targets.add(adapter_name)
        return targets

    @property
    def tag_routes(self) -> dict[str, RoutingRule]:
        """Current tag routing rules (read-only copy)."""
        return dict(self._tag_routes)

    def describe(self) -> dict:
        """Describe current routing state for `praxis logger status`."""
        return {
            "critical_override": self.critical_override,
            "tag_routes": {
                tag: {
                    "adapters": rule.adapter_names,
                    "min_level": rule.min_level,
                }
                for tag, rule in self._tag_routes.items()
            },
        }
