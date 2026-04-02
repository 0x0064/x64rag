function AnalyzePage(page_image: image) -> PageAnalysis {
  client Default
  prompt #"
    You are a technical document analyst. Analyze this page image and extract all structured information.

    Instructions:
    - Describe what the page contains (schematic, table, diagram, text, etc.)
    - Extract all identifiable entities (components, parts, valves, motors, etc.) with their categories
    - Extract all tables with their column headers and row data
    - Note any annotations, callouts, or reference marks
    - Identify the page type (schematic, bom_table, wiring_diagram, assembly, detail, p_and_id, text, etc.)

    ======== IMPORTANT ========
    - Extract what you actually see — do NOT invent or hallucinate values
    - If a value is unclear, use the entity's context field to note uncertainty
    - Categories should be descriptive (component, valve, motor, wire, terminal, dimension, etc.)
    ======== END IMPORTANT ========

    {{ ctx.output_format }}
  "#
}
