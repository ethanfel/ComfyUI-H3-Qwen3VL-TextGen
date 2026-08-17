import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / "example_workflows"
    / "h3_qwen_vl_standalone_polaris_overlay.json"
)
OVERLAY = (
    "MiniMax H3/minimax_h3_qwen3vl32b__prompt_generation_overlay__"
    "polaris_r16_plus_heretic_v2.safetensors"
)


def test_example_workflow_uses_matching_h3_models_and_overlay():
    workflow = json.loads(WORKFLOW.read_text())
    nodes = {node["type"]: node for node in workflow["nodes"]}

    assert nodes["CLIPLoader"]["widgets_values"] == [
        "MiniMax-H3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "minimax",
        "default",
    ]
    assert nodes["H3QwenVLGenerationTailLoader"]["widgets_values"] == [
        "MiniMax-H3/"
        "qwen3vl_32b_h3_instruct_generation_tail_50_63_int8_convrot.safetensors"
    ]

    generator = nodes["H3QwenVLGenerateText"]
    input_names = [item["name"] for item in generator["inputs"]]
    assert input_names[:4] == [
        "clip",
        "tail_clip",
        "generation_overlay",
        "overlay_strength",
    ]
    assert generator["widgets_values"][:2] == [OVERLAY, 1.0]

    urls = [
        model["url"]
        for node in workflow["nodes"]
        for model in node.get("properties", {}).get("models", [])
    ]
    assert len(urls) == 3
    assert all(url.startswith("https://huggingface.co/") for url in urls)


def test_example_workflow_links_reference_existing_nodes_and_slots():
    workflow = json.loads(WORKFLOW.read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    for link_id, source_id, source_slot, target_id, target_slot, _ in links.values():
        assert link_id in links
        assert source_id in nodes
        assert target_id in nodes
        assert source_slot < len(nodes[source_id]["outputs"])
        assert target_slot < len(nodes[target_id]["inputs"])
