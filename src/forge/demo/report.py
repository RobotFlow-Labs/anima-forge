"""HTML report generator with SVG diagrams and charts.

Generates a single self-contained HTML file with:
- Hero section: FORGE v3 headline + measured checkpoint numbers
- Architecture diagram (SVG)
- Benchmark results (bar charts via inline SVG)
- Teacher comparison table
- Embodiment profiles
- Compression pipeline visualization
"""

from __future__ import annotations

import math


def _metric(mapping: dict, key: str, suffix: str = "") -> str:
    """Format a finite measured metric without substituting a marketing default."""
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.0f}{suffix}"


def generate_html_report(data: dict) -> str:
    """Generate complete HTML report from demo data."""

    benchmark = data.get("benchmark", {})
    teachers = data.get("teachers", [])
    embodiments = data.get("embodiments", [])
    provenance = data.get("provenance", {})
    component_keys = ("vision", "language", "labels")
    missing_components = [key for key in component_keys if provenance.get(key) not in {"real", "mock"}]
    mock_components = [key for key in component_keys if provenance.get(key) == "mock"]
    if missing_components or mock_components:
        detail = (
            "Missing provenance: " + ", ".join(missing_components)
            if missing_components
            else "Mock provenance: " + ", ".join(mock_components)
        )
        provenance_banner = (
            """<div style="background:#5c1010;color:#fff;padding:1rem;text-align:center;font-weight:700;">"""
            "[MOCK — not a real model] " + detail + "</div>"
        )
    else:
        provenance_banner = (
            """<div style="background:#0b4f36;color:#fff;padding:1rem;text-align:center;font-weight:700;">"""
            "REAL provenance verified"
            "</div>"
        )

    latency = benchmark.get("latency", {})
    throughput = benchmark.get("throughput", {})
    compression = benchmark.get("compression", {})
    input_provenance = benchmark.get("input_provenance", {})
    if isinstance(input_provenance, dict) and input_provenance.get("kind") == "real":
        input_banner = (
            '<div style="background:#0b4f36;color:#fff;padding:0.75rem;text-align:center;font-weight:700;">'
            "REAL benchmark input provenance verified</div>"
        )
    else:
        input_banner = (
            '<div style="background:#5c1010;color:#fff;padding:0.75rem;text-align:center;font-weight:700;">'
            "Benchmark input is not verified as real; metrics are not launch evidence</div>"
        )

    # Build teacher rows
    if teachers:
        teacher_rows = "".join(
            f"<tr><td>{t.get('name', '')}</td>"
            f"<td>{t.get('architecture', '')}</td>"
            f"<td>{t.get('params_b', 0):.1f}B</td>"
            f"<td>{'Yes' if t.get('supports_chunking') else 'No'}</td></tr>"
            for t in teachers
        )
    else:
        teacher_rows = '<tr><td colspan="4">No teacher registry metadata was available for this report.</td></tr>'

    # Build embodiment rows
    if embodiments:
        embodiment_rows = "".join(
            f"<tr><td>{e.get('name', '')}</td><td>{e.get('dof', 0)}-DoF</td><td>Auto-configured</td></tr>"
            for e in embodiments
        )
    else:
        embodiment_rows = '<tr><td colspan="3">No embodiment registry metadata was available for this report.</td></tr>'

    arch_svg = _generate_architecture_svg()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FORGE v3 — VLA Distillation and Deployment Report</title>
    <style>
        :root {{
            --bg: #0a0a0f;
            --surface: #12121a;
            --border: #1e1e2e;
            --text: #e0e0e8;
            --muted: #888899;
            --accent: #00d4ff;
            --accent2: #ff6b35;
            --accent3: #00ff88;
            --gradient: linear-gradient(135deg, #00d4ff, #00ff88);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}

        /* Hero */
        .hero {{
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(180deg, var(--surface) 0%, var(--bg) 100%);
            border-bottom: 1px solid var(--border);
        }}
        .hero h1 {{
            font-size: 3rem;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }}
        .hero .subtitle {{ color: var(--muted); font-size: 1.2rem; }}

        /* Key numbers */
        .numbers {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 3rem 0;
        }}
        .number-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }}
        .number-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--accent);
        }}
        .number-card .label {{
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}

        /* Sections */
        .section {{
            margin: 3rem 0;
            padding: 2rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }}
        .section h2 {{
            color: var(--accent);
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ color: var(--accent); font-size: 0.85rem; text-transform: uppercase; }}

        /* Architecture diagram */
        .arch-diagram {{
            text-align: center;
            padding: 2rem;
        }}
        .arch-diagram svg {{
            max-width: 100%;
        }}

        /* Bar chart */
        .bar-chart {{
            display: flex;
            align-items: end;
            gap: 2rem;
            height: 200px;
            padding: 1rem 0;
        }}
        .bar {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .bar-fill {{
            width: 60px;
            background: var(--gradient);
            border-radius: 6px 6px 0 0;
            transition: height 0.3s;
        }}
        .bar-label {{
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }}
        .bar-value {{
            color: var(--accent);
            font-weight: bold;
            margin-bottom: 0.25rem;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--muted);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            margin-top: 3rem;
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>FORGE v3</h1>
        <div class="subtitle">VLA Distillation and Deployment Report</div>
        <p style="color: var(--muted); margin-top: 1rem;">
            Checkpoint-backed inference, compression, and deployment evidence
        </p>
    </div>
    {provenance_banner}
    {input_banner}

    <div class="container">
        <!-- Key Numbers -->
        <div class="numbers">
            <div class="number-card">
                <div class="value">{_metric(compression, "compression_ratio", "x")}</div>
                <div class="label">Compression Ratio</div>
            </div>
            <div class="number-card">
                <div class="value">{_metric(latency, "mean_ms", "ms")}</div>
                <div class="label">Inference Latency ({benchmark.get("device", "unknown device")})</div>
            </div>
            <div class="number-card">
                <div class="value">{_metric(throughput, "actions_per_second")}</div>
                <div class="label">Actions/sec (chunked)</div>
            </div>
            <div class="number-card">
                <div class="value">{_metric(compression, "model_size_mb", "MB")}</div>
                <div class="label">Model Size (INT4)</div>
            </div>
            <div class="number-card">
                <div class="value">{_metric(throughput, "chunk_gain", "x")}</div>
                <div class="label">Chunk Throughput Gain</div>
            </div>
            <div class="number-card">
                <div class="value">3</div>
                <div class="label">Teacher Models</div>
            </div>
        </div>

        <!-- Architecture -->
        <div class="section">
            <h2>Architecture</h2>
            <div class="arch-diagram">
                {arch_svg}
            </div>
        </div>

        <!-- Teachers -->
        <div class="section">
            <h2>Universal Teacher Support</h2>
            <table>
                <thead>
                    <tr><th>Teacher</th><th>Architecture</th><th>Parameters</th><th>Chunking</th></tr>
                </thead>
                <tbody>
                    {teacher_rows}
                </tbody>
            </table>
        </div>

        <!-- Embodiments -->
        <div class="section">
            <h2>Robot Embodiment Profiles</h2>
            <table>
                <thead>
                    <tr><th>Robot</th><th>DoF</th><th>Recommended Config</th></tr>
                </thead>
                <tbody>
                    {embodiment_rows}
                </tbody>
            </table>
        </div>

        <!-- Pipeline -->
        <div class="section">
            <h2>Distillation Pipeline</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; text-align: center;">
                <div style="padding: 1rem; border: 1px solid var(--accent); border-radius: 8px;">
                    <div style="color: var(--accent); font-weight: bold;">Stage 1</div>
                    <div>Multi-Teacher Labels</div>
                    <div style="color: var(--muted); font-size: 0.8rem;">OpenVLA + RDT2 + SmolVLA</div>
                </div>
                <div style="padding: 1rem; border: 1px solid var(--accent); border-radius: 8px;">
                    <div style="color: var(--accent); font-weight: bold;">Stage 2</div>
                    <div>Multi-Path KD</div>
                    <div style="color: var(--muted); font-size: 0.8rem;">Learned teacher routing</div>
                </div>
                <div style="padding: 1rem; border: 1px solid var(--accent); border-radius: 8px;">
                    <div style="color: var(--accent); font-weight: bold;">Stage 3</div>
                    <div>Chunk Compression</div>
                    <div style="color: var(--muted); font-size: 0.8rem;">Temporal-aware pruning + INT4</div>
                </div>
                <div style="padding: 1rem; border: 1px solid var(--accent); border-radius: 8px;">
                    <div style="color: var(--accent); font-weight: bold;">Stage 4</div>
                    <div>Edge Export</div>
                    <div style="color: var(--muted); font-size: 0.8rem;">TensorRT / CoreML / MLX</div>
                </div>
            </div>
        </div>

        <div class="footer">
            FORGE v3 &mdash; RobotFlowLabs<br>
            Part of the ANIMA Agentic Robotics AI Stack
        </div>
    </div>
</body>
</html>"""

    return html


def _generate_architecture_svg() -> str:
    """Generate SVG architecture diagram."""
    return """<svg viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg">
    <!-- Teacher -->
    <rect x="20" y="20" width="160" height="80" rx="8" fill="#1a1a2e" stroke="#00d4ff" stroke-width="1.5"/>
    <text x="100" y="55" text-anchor="middle" fill="#00d4ff" font-size="14" font-family="monospace">Teacher VLA</text>
    <text x="100" y="80" text-anchor="middle" fill="#888" font-size="11" font-family="monospace">7B+ params</text>

    <!-- Arrow -->
    <path d="M 180 60 L 240 60" stroke="#00d4ff" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
    <text x="210" y="50" text-anchor="middle" fill="#888" font-size="10" font-family="monospace">KD</text>

    <!-- Student -->
    <rect x="240" y="20" width="160" height="80" rx="8" fill="#1a1a2e" stroke="#00ff88" stroke-width="1.5"/>
    <text x="320" y="55" text-anchor="middle" fill="#00ff88" font-size="14" font-family="monospace">FORGE Student</text>
    <text x="320" y="80" text-anchor="middle" fill="#888" font-size="11" font-family="monospace">0.5B params</text>

    <!-- Arrow -->
    <path d="M 400 60 L 460 60" stroke="#00ff88" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>

    <!-- Compress -->
    <rect x="460" y="20" width="160" height="80" rx="8" fill="#1a1a2e" stroke="#ff6b35" stroke-width="1.5"/>
    <text x="540" y="55" text-anchor="middle" fill="#ff6b35" font-size="14" font-family="monospace">Compressed</text>
    <text x="540" y="80" text-anchor="middle" fill="#888" font-size="11" font-family="monospace">INT4 + Pruned</text>

    <!-- Arrow -->
    <path d="M 620 60 L 680 60" stroke="#ff6b35" stroke-width="1.5" fill="none" marker-end="url(#arrow3)"/>

    <!-- Edge -->
    <rect x="680" y="20" width="100" height="80" rx="8" fill="#1a1a2e" stroke="#fff" stroke-width="1.5"/>
    <text x="730" y="55" text-anchor="middle" fill="#fff" font-size="14" font-family="monospace">Edge</text>
    <text x="730" y="80" text-anchor="middle" fill="#888" font-size="11" font-family="monospace">&lt;500MB</text>

    <!-- Features row -->
    <rect x="20" y="140" width="760" height="40" rx="6" fill="#0d0d15" stroke="#1e1e2e"/>
    <text x="100" y="165" text-anchor="middle" fill="#00d4ff" font-size="11"
          font-family="monospace">Multi-Teacher</text>
    <text x="280" y="165" text-anchor="middle" fill="#00ff88" font-size="11"
          font-family="monospace">Action Chunking (H=8)</text>
    <text x="460" y="165" text-anchor="middle" fill="#ff6b35" font-size="11"
          font-family="monospace">Flow Matching (1-step)</text>
    <text x="660" y="165" text-anchor="middle" fill="#fff" font-size="11" font-family="monospace">Async Runtime</text>

    <!-- Arrow markers -->
    <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#00d4ff"/>
        </marker>
        <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#00ff88"/>
        </marker>
        <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff6b35"/>
        </marker>
    </defs>
</svg>"""
