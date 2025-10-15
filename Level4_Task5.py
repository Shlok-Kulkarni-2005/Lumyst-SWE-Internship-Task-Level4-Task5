"""
Utility Function Detection and Filtering

Detects and ranks functions by likely business relevance using lightweight
static analysis and simple heuristics. No ML/LLM. Accepts the FastAPI analysis
JSON (nodes with id/label/code) and returns a ranked list with categories.
"""

import json
import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum


class FunctionCategory(Enum):
    """Classification of function types by business relevance."""
    CORE_LOGIC = 5.0        # Business logic, domain operations
    DATA_PROCESSING = 4.0   # Data transformation, enrichment
    INFRASTRUCTURE = 2.5    # Config, setup, initialization
    UTILITY = 1.0          # Helpers, formatters, parsers
    TRIVIAL = 0.0          # One-liners, type checks, simple getters


@dataclass
class FunctionMetrics:
    """Metrics computed for a function during analysis."""
    cyclomatic_complexity: int = 0
    code_lines: int = 0
    has_side_effects: bool = False
    calls_count: int = 0
    called_by_count: int = 0
    parameters_count: int = 0
    returns_complex_type: bool = False
    has_docstring: bool = False
    is_generic_helper: bool = False
    naming_score: float = 0.0
    

@dataclass
class FunctionNode:
    """Represents a function in the analysis graph."""
    id: str
    label: str
    code: str
    file: str
    metrics: FunctionMetrics = field(default_factory=FunctionMetrics)
    category: FunctionCategory = FunctionCategory.UTILITY
    importance_score: float = 0.0
    is_utility: bool = False
    reasoning: str = ""


class UtilityFunctionDetector:
    """
    Static analysis engine for detecting utility functions.
    
    Approach:
    1. Extract code metrics (complexity, LOC, patterns)
    2. Analyze naming conventions and documentation
    3. Compute call graph relationships
    4. Apply heuristic scoring rules
    5. Classify and rank functions
    """
    
    # Utility/generic naming patterns
    UTILITY_PATTERNS = {
        r'^(get|set|is|has|format|parse|encode|decode|convert|transform)_',
        r'^(to|from|as)_',
        r'^_(private|helper)',
        r'.*_(util|helper|common|utils)$',
        r'^(validate|check|verify|sanitize)_',
        r'^(log|logger|logging|print|debug|trace)',
        r'^(round|ceil|floor|abs|min|max|sum|avg|mean|median)_',
        r'^(date|time|datetime|timestamp|isoformat|strptime|strftime|utcnow|now)_',
        r'^(open|close|read|write)_',
    }
    
    # Generic utility library functions
    GENERIC_FUNCTIONS = {
        'format', 'parse', 'encode', 'decode', 'stringify', 'convert',
        'validate', 'check', 'verify', 'filter', 'map', 'reduce',
        'sort', 'reverse', 'clone', 'copy', 'merge', 'combine',
        'split', 'join', 'trim', 'strip', 'capitalize', 'lowercase',
        # Logging helpers
        'log', 'logger', 'debug', 'trace',
        # Math helpers
        'round', 'ceil', 'floor', 'abs', 'min', 'max', 'sum',
        # Datetime helpers
        'now', 'utcnow', 'today', 'strftime', 'strptime',
    }
    
    # Core business keywords
    BUSINESS_KEYWORDS = {
        'create', 'update', 'delete', 'save', 'process', 'handle',
        'calculate', 'compute', 'analyze', 'report', 'fetch', 'search',
        'query', 'execute', 'run', 'start', 'initialize', 'setup',
    }

    def __init__(self):
        self.function_map: Dict[str, FunctionNode] = {}
        self.call_graph: Dict[str, Set[str]] = {}
        
    def analyze(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for analysis."""
        nodes = graph_data.get('analysisData', {}).get('graphNodes', [])
        
        # First pass: extract metrics
        for node_data in nodes:
            node = self._extract_node(node_data)
            self.function_map[node.id] = node
            self._compute_metrics(node)
        
        # Second pass: build call graph and analyze relationships
        self._build_call_graph()
        self._update_call_metrics()
        
        # Third pass: score and classify
        for node in self.function_map.values():
            self._score_function(node)
            node.category = self._classify_function(node)
        
        # Generate output
        return self._generate_output()
    
    def _extract_node(self, node_data: Dict[str, Any]) -> FunctionNode:
        """Extract basic node information."""
        return FunctionNode(
            id=node_data.get('id', ''),
            label=node_data.get('label', ''),
            code=node_data.get('code', ''),
            file=node_data.get('id', '').split(':')[1] if ':' in node_data.get('id', '') else '',
        )
    
    def _compute_metrics(self, node: FunctionNode) -> None:
        """Static analysis of function code."""
        code = node.code
        lines = code.split('\n')
        
        # Basic metrics
        node.metrics.code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        node.metrics.has_docstring = '"""' in code or "'''" in code
        node.metrics.parameters_count = self._count_parameters(code)
        
        # Complexity: count conditionals and loops
        node.metrics.cyclomatic_complexity = self._calculate_complexity(code)
        
        # Side effects: mutations, external calls
        node.metrics.has_side_effects = self._has_side_effects(code)
        
        # Complex return types
        node.metrics.returns_complex_type = self._returns_complex_type(code)
        
        # Call count
        node.metrics.calls_count = len(re.findall(r'\w+\(', code))
        
        # Generic helper detection
        node.metrics.is_generic_helper = self._is_generic_helper(node.label)
        
        # Naming score (0-1, higher is more business-like)
        node.metrics.naming_score = self._score_name(node.label)
    
    def _count_parameters(self, code: str) -> int:
        """Count function parameters."""
        match = re.search(r'def\s+\w+\((.*?)\):', code)
        if match:
            params = match.group(1)
            # Simple count: split by comma, filter out empty
            return len([p.strip() for p in params.split(',') if p.strip()])
        return 0
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1
        keywords = ['if', 'elif', 'else', 'for', 'while', 'except', 'and', 'or']
        for keyword in keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', code))
        return complexity
    
    def _has_side_effects(self, code: str) -> bool:
        """Detect if function has side effects (mutations, I/O, etc)."""
        patterns = [
            r'\.set\(',
            r'\.append\(',
            r'\.update\(',
            r'\.write\(',
            r'\.save\(',
            r'self\.\w+\s*=',
            r'global\s+',
        ]
        for pattern in patterns:
            if re.search(pattern, code):
                return True
        return False
    
    def _returns_complex_type(self, code: str) -> bool:
        """Check if function returns complex/domain types."""
        return bool(
            re.search(r'->\s*.*(?:Dict|List|Tuple|Set|Model|Entity|Schema|Response|Optional\[)', code)
        )
    
    def _is_generic_helper(self, name: str) -> bool:
        """Check if name indicates a generic utility function."""
        name_lower = name.lower()
        for pattern in self.UTILITY_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        return name_lower in self.GENERIC_FUNCTIONS
    
    def _score_name(self, name: str) -> float:
        """Score function name for business relevance (0-1)."""
        name_lower = name.lower()
        
        # Business keywords boost score
        for keyword in self.BUSINESS_KEYWORDS:
            if keyword in name_lower:
                return 0.8
        
        # Generic utility names lower score
        for pattern in self.UTILITY_PATTERNS:
            if re.search(pattern, name_lower):
                return 0.2
        
        # Default middle ground
        return 0.5
    
    def _build_call_graph(self) -> None:
        """Extract function call relationships."""
        for node_id, node in self.function_map.items():
            calls = set()
            # Extract function calls from code
            for func_name in re.findall(r'([a-zA-Z_]\w*)\(', node.code):
                # Try to match to known functions
                for other_id, other_node in self.function_map.items():
                    if func_name == other_node.label or func_name in other_id:
                        calls.add(other_id)
            self.call_graph[node_id] = calls
    
    def _update_call_metrics(self) -> None:
        """Update metrics based on call relationships."""
        # Count how many times each function is called
        called_count = {node_id: 0 for node_id in self.function_map}
        
        for caller_id, callees in self.call_graph.items():
            for callee_id in callees:
                if callee_id in called_count:
                    called_count[callee_id] += 1
        
        for node_id, node in self.function_map.items():
            node.metrics.called_by_count = called_count.get(node_id, 0)
    
    def _score_function(self, node: FunctionNode) -> None:
        """Compute importance score using multiple factors."""
        score = 0.0
        
        # Factor 1: Naming and patterns (0-30 points)
        score += node.metrics.naming_score * 30
        if node.metrics.is_generic_helper:
            score -= 15
        
        # Factor 2: Code complexity (0-25 points)
        # Higher complexity = more likely business logic
        complexity_score = min(node.metrics.cyclomatic_complexity / 10, 1.0)
        score += complexity_score * 25
        
        # Factor 3: Side effects (0-20 points)
        # Functions with side effects are often business critical
        if node.metrics.has_side_effects:
            score += 20
        
        # Factor 4: Return types (0-15 points)
        if node.metrics.returns_complex_type:
            score += 15
        
        # Factor 5: Call graph position (0-10 points)
        # Frequently called functions are more important
        if node.metrics.called_by_count > 2:
            score += min(node.metrics.called_by_count * 2, 10)
        
        # Factor 6: Documentation (0-5 points)
        if node.metrics.has_docstring:
            score += 5
        
        # Normalize to 0-100
        node.importance_score = max(0, min(100, score))
    
    def _classify_function(self, node: FunctionNode) -> FunctionCategory:
        """Classify function into a category."""
        score = node.importance_score
        
        # Decision tree
        if node.metrics.code_lines < 2 and node.metrics.cyclomatic_complexity <= 1:
            return FunctionCategory.TRIVIAL
        
        if node.metrics.is_generic_helper:
            return FunctionCategory.UTILITY
        
        if 'init' in node.label.lower() or 'setup' in node.label.lower():
            return FunctionCategory.INFRASTRUCTURE
        
        if node.metrics.has_side_effects and score > 50:
            return FunctionCategory.CORE_LOGIC
        
        if node.metrics.returns_complex_type and score > 40:
            return FunctionCategory.DATA_PROCESSING
        
        if score > 60:
            return FunctionCategory.CORE_LOGIC
        elif score > 40:
            return FunctionCategory.DATA_PROCESSING
        elif score > 20:
            return FunctionCategory.INFRASTRUCTURE
        else:
            return FunctionCategory.UTILITY
    
    def _generate_output(self) -> Dict[str, Any]:
        """Generate structured output."""
        ranked_functions = sorted(
            self.function_map.values(),
            key=lambda n: n.importance_score,
            reverse=True
        )
        
        output = {
            'summary': {
                'total_functions': len(self.function_map),
                'core_logic': len([n for n in self.function_map.values() if n.category == FunctionCategory.CORE_LOGIC]),
                'utilities': len([n for n in self.function_map.values() if n.category == FunctionCategory.UTILITY]),
                'trivial': len([n for n in self.function_map.values() if n.category == FunctionCategory.TRIVIAL]),
            },
            'ranked_functions': [
                {
                    'id': node.id,
                    'name': node.label,
                    'importance_score': round(node.importance_score, 2),
                    'category': node.category.name,
                    'is_utility': node.category in [FunctionCategory.UTILITY, FunctionCategory.TRIVIAL],
                    'metrics': {
                        'lines_of_code': node.metrics.code_lines,
                        'complexity': node.metrics.cyclomatic_complexity,
                        'called_by_count': node.metrics.called_by_count,
                        'has_side_effects': node.metrics.has_side_effects,
                    }
                }
                for node in ranked_functions
            ]
        }
        
        return output


def filter_utility_functions(json_input: str) -> Dict[str, Any]:
    """
    Main entry point. Accepts JSON and returns filtered analysis.
    """
    data = json.loads(json_input)
    detector = UtilityFunctionDetector()
    return detector.analyze(data)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Detect and rank utility vs business logic functions from analysis JSON"
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to JSON file produced by your analysis pipeline. Defaults to stdin if omitted.",
    )
    parser.add_argument(
        "--business-only",
        action="store_true",
        help="Only output non-utility functions (exclude UTILITY and TRIVIAL)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum importance_score to include in output",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    try:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                raw = f.read()
        else:
            raw = sys.stdin.read()

        result = filter_utility_functions(raw)

        if args.business_only or args.min_score is not None:
            filtered = []
            for item in result.get("ranked_functions", []):
                if args.business_only and item.get("is_utility"):
                    continue
                if args.min_score is not None and float(item.get("importance_score", 0)) < args.min_score:
                    continue
                filtered.append(item)
            result = {
                "summary": result.get("summary", {}),
                "ranked_functions": filtered,
            }

        print(json.dumps(result, indent=2 if args.pretty else None))
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())