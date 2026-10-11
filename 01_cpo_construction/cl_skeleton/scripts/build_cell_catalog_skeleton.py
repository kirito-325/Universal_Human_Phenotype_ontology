#!/usr/bin/env python3
"""Build a compact cell catalog skeleton for review."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBO_PATH = ROOT / "cell_ontology" / "cl-basic.obo"
OUT_DIR = ROOT / "cell_ontology" / "skeleton"


@dataclass(frozen=True)
class Node:
    cl_id: str
    parent_id: str
    branch: str
    rank: str
    label: str
    label_zh: str
    note: str


LEGACY_AI_SKELETON = [
    Node("CL:0000000", "", "root", "root", "Cell", "细胞", "Root of the catalog."),
    Node("CL:0011115", "CL:0000000", "stem_progenitor", "branch", "Stem and progenitor cells", "干细胞与祖细胞", "Cross-lineage immature cell states."),
    Node("CL:0000034", "CL:0011115", "stem_progenitor", "group", "Stem cells", "干细胞", "Generic stem cell bucket."),
    Node("CL:0011026", "CL:0011115", "stem_progenitor", "type", "Progenitor cell", "祖细胞", "Generic progenitor bucket."),
    Node("CL:0002322", "CL:0000034", "stem_progenitor", "type", "Embryonic stem cell", "胚胎干细胞", "Keep as a broad stem-cell subtype."),
    Node("CL:0000134", "CL:0000034", "stem_progenitor", "type", "Mesenchymal stem cell", "间充质干细胞", "Often used as a catalog-level cell type."),
    Node("CL:0002321", "CL:0000000", "developmental", "branch", "Embryonic and developmental cells", "胚胎与发育细胞", "Development-stage cells that are not mature tissue cell types."),
    Node("CL:0000349", "CL:0002321", "developmental", "type", "Extraembryonic cell", "胚外细胞", "Developmental branch."),
    Node("CL:0010017", "CL:0002321", "developmental", "type", "Zygote", "合子", "Developmental branch."),
    Node("CL:0000066", "CL:0000000", "epithelial", "branch", "Epithelial cells", "上皮细胞", "Barrier, lining, glandular, absorptive, and secretory epithelial cells."),
    Node("CL:0000646", "CL:0000066", "epithelial", "type", "Basal cell", "基底细胞", "Basal/reserve epithelial cell."),
    Node("CL:0000312", "CL:0000066", "epithelial", "type", "Keratinocyte", "角质形成细胞", "Skin and stratified epithelial cell."),
    Node("CL:0000067", "CL:0000066", "epithelial", "type", "Ciliated epithelial cell", "纤毛上皮细胞", "Ciliated epithelial cell."),
    Node("CL:1100001", "CL:0000066", "epithelial", "group", "Secretory epithelial cells", "分泌型上皮细胞", "Glandular and mucosal epithelial secretory cells."),
    Node("CL:0000160", "CL:1100001", "epithelial", "type", "Goblet cell", "杯状细胞", "Canonical mucous secretory epithelial cell."),
    Node("CL:0000158", "CL:1100001", "epithelial", "type", "Club cell", "Club 细胞", "Respiratory secretory epithelial cell."),
    Node("CL:0002563", "CL:0000066", "epithelial", "group", "Intestinal epithelial cells", "肠上皮细胞", "Digestive tract epithelial branch."),
    Node("CL:0000584", "CL:0002563", "epithelial", "type", "Enterocyte", "肠吸收细胞", "Canonical absorptive intestinal epithelial cell."),
    Node("CL:0002518", "CL:0000066", "epithelial", "group", "Kidney epithelial cells", "肾上皮细胞", "Kidney epithelial branch."),
    Node("CL:0000653", "CL:0002518", "epithelial", "type", "Podocyte", "足细胞", "Glomerular epithelial cell."),
    Node("CL:0005009", "CL:0002518", "epithelial", "type", "Renal principal cell", "肾主细胞", "Collecting duct epithelial branch."),
    Node("CL:0005010", "CL:0002518", "epithelial", "type", "Renal intercalated cell", "肾闰细胞", "Collecting duct epithelial branch."),
    Node("CL:0000182", "CL:0000066", "epithelial", "type", "Hepatocyte", "肝细胞", "Major liver parenchymal epithelial cell."),
    Node("CL:1000488", "CL:0000066", "epithelial", "type", "Cholangiocyte", "胆管细胞", "Biliary epithelial cell."),
    Node("CL:0002062", "CL:0000066", "epithelial", "type", "Alveolar type 1 cell", "肺泡 I 型细胞", "Lung epithelial branch."),
    Node("CL:0002063", "CL:0000066", "epithelial", "type", "Alveolar type 2 cell", "肺泡 II 型细胞", "Lung epithelial branch."),
    Node("CL:0000115", "CL:0000000", "endothelial", "branch", "Endothelial cells", "内皮细胞", "Vascular and lymphatic lining cells."),
    Node("CL:0002139", "CL:0000115", "endothelial", "group", "Vascular endothelial cells", "血管内皮细胞", "Blood vascular endothelial compartment."),
    Node("CL:0000071", "CL:0002139", "endothelial", "type", "Blood vessel endothelial cell", "血管内皮细胞", "General blood-vessel endothelial cell."),
    Node("CL:2000008", "CL:0002139", "endothelial", "type", "Microvascular endothelial cell", "微血管内皮细胞", "Capillary/microvascular endothelial branch."),
    Node("CL:0002138", "CL:0000115", "endothelial", "type", "Lymphatic endothelial cell", "淋巴管内皮细胞", "Lymphatic endothelial cell."),
    Node("CL:0002320", "CL:0000000", "stromal_connective", "branch", "Stromal and connective tissue cells", "基质与结缔组织细胞", "Support, matrix-producing, skeletal, and perivascular cells."),
    Node("CL:0000499", "CL:0002320", "stromal_connective", "group", "Stromal cells", "基质细胞", "General tissue stromal bucket."),
    Node("CL:0000057", "CL:0000499", "stromal_connective", "type", "Fibroblast", "成纤维细胞", "Canonical matrix-producing stromal cell."),
    Node("CL:0008019", "CL:0002320", "stromal_connective", "type", "Mesenchymal cell", "间充质细胞", "Broad mesenchymal cell."),
    Node("CL:0008034", "CL:0002320", "stromal_connective", "type", "Mural cell", "血管壁细胞", "Perivascular contractile/support cell."),
    Node("CL:0000669", "CL:0002320", "stromal_connective", "type", "Pericyte", "周细胞", "Canonical mural/perivascular cell."),
    Node("CL:0000136", "CL:0002320", "stromal_connective", "type", "Adipocyte", "脂肪细胞", "Adipose-lineage mature cell."),
    Node("CL:0001035", "CL:0002320", "stromal_connective", "group", "Bone cells", "骨细胞谱系", "Bone remodeling and matrix branch."),
    Node("CL:0000062", "CL:0001035", "stromal_connective", "type", "Osteoblast", "成骨细胞", "Bone-forming cell."),
    Node("CL:0000137", "CL:0001035", "stromal_connective", "type", "Osteocyte", "骨细胞", "Mature bone cell."),
    Node("CL:0000092", "CL:0001035", "stromal_connective", "type", "Osteoclast", "破骨细胞", "Bone-resorbing cell; hematopoietic origin but cataloged here by tissue role."),
    Node("CL:0000138", "CL:0002320", "stromal_connective", "type", "Chondrocyte", "软骨细胞", "Cartilage cell."),
    Node("CL:0000187", "CL:0000000", "muscle", "branch", "Muscle cells", "肌肉细胞", "Contractile muscle and myogenic lineage cells."),
    Node("CL:0000192", "CL:0000187", "muscle", "type", "Smooth muscle cell", "平滑肌细胞", "Catalog-level smooth muscle cell."),
    Node("CL:0000359", "CL:0000187", "muscle", "type", "Vascular smooth muscle cell", "血管平滑肌细胞", "Catalog-level vascular smooth muscle cell."),
    Node("CL:0008002", "CL:0000187", "muscle", "type", "Skeletal muscle fiber", "骨骼肌纤维", "Mature skeletal muscle cell."),
    Node("CL:0002494", "CL:0000187", "muscle", "type", "Cardiocyte", "心肌细胞谱系", "Catalog-level cardiac cell."),
    Node("CL:0000746", "CL:0000187", "muscle", "type", "Cardiac muscle cell", "心肌细胞", "Canonical mature cardiac muscle cell."),
    Node("CL:0000056", "CL:0000187", "muscle", "type", "Myoblast", "成肌细胞", "Myogenic precursor."),
    Node("CL:0002319", "CL:0000000", "neural", "branch", "Neural cells", "神经系统细胞", "Neurons, glia, and neural progenitors."),
    Node("CL:0011020", "CL:0002319", "neural", "type", "Neural progenitor cell", "神经祖细胞", "Immature neural lineage cell."),
    Node("CL:0000540", "CL:0002319", "neural", "group", "Neurons", "神经元", "Neuronal branch."),
    Node("CL:0000101", "CL:0000540", "neural", "type", "Sensory neuron", "感觉神经元", "Functional neuronal category."),
    Node("CL:0000100", "CL:0000540", "neural", "type", "Motor neuron", "运动神经元", "Functional neuronal category."),
    Node("CL:0000099", "CL:0000540", "neural", "type", "Interneuron", "中间神经元", "Circuit-level neuronal category."),
    Node("CL:0000617", "CL:0000540", "neural", "type", "GABAergic neuron", "GABA 能神经元", "Neurotransmitter-defined neuronal category."),
    Node("CL:0000679", "CL:0000540", "neural", "type", "Glutamatergic neuron", "谷氨酸能神经元", "Neurotransmitter-defined neuronal category."),
    Node("CL:0000700", "CL:0000540", "neural", "type", "Dopaminergic neuron", "多巴胺能神经元", "Neurotransmitter-defined neuronal category."),
    Node("CL:0000125", "CL:0002319", "neural", "group", "Glial cells", "胶质细胞", "Non-neuronal neural support cells."),
    Node("CL:0000127", "CL:0000125", "neural", "type", "Astrocyte", "星形胶质细胞", "Major CNS glial type."),
    Node("CL:0000128", "CL:0000125", "neural", "type", "Oligodendrocyte", "少突胶质细胞", "Myelinating CNS glial cell."),
    Node("CL:0002453", "CL:0000125", "neural", "type", "Oligodendrocyte precursor cell", "少突胶质前体细胞", "Common scRNA annotation class."),
    Node("CL:0000129", "CL:0000125", "neural", "type", "Microglial cell", "小胶质细胞", "CNS-resident immune/glial cell."),
    Node("CL:0002573", "CL:0000125", "neural", "type", "Schwann cell", "施万细胞", "Peripheral glial cell."),
    Node("CL:0000065", "CL:0000125", "neural", "type", "Ependymal cell", "室管膜细胞", "Ventricular/CSF-associated neural cell."),
    Node("CL:0000988", "CL:0000000", "blood_immune", "branch", "Blood and immune cells", "血液与免疫细胞", "Hematopoietic, immune, erythroid, and platelet branches."),
    Node("CL:0008001", "CL:0000988", "blood_immune", "group", "Hematopoietic precursor cells", "造血前体细胞", "Immature hematopoietic branch."),
    Node("CL:0000037", "CL:0008001", "blood_immune", "type", "Hematopoietic stem cell", "造血干细胞", "Stem cell of hematopoietic lineage."),
    Node("CL:0000081", "CL:0000988", "blood_immune", "group", "Blood cells", "血细胞", "Circulating blood cell branch."),
    Node("CL:0000232", "CL:0000081", "blood_immune", "type", "Erythrocyte", "红细胞", "Red blood cell."),
    Node("CL:0000556", "CL:0000081", "blood_immune", "type", "Megakaryocyte", "巨核细胞", "Platelet-producing lineage cell."),
    Node("CL:0000233", "CL:0000081", "blood_immune", "type", "Platelet", "血小板", "Anucleate blood element."),
    Node("CL:0000738", "CL:0000988", "blood_immune", "group", "Leukocytes", "白细胞", "Immune cell branch."),
    Node("CL:0000542", "CL:0000738", "blood_immune", "group", "Lymphocytes", "淋巴细胞", "Lymphoid immune branch."),
    Node("CL:0000084", "CL:0000542", "blood_immune", "type", "T cell", "T 细胞", "Major lymphocyte branch."),
    Node("CL:0000236", "CL:0000542", "blood_immune", "type", "B cell", "B 细胞", "Major lymphocyte branch."),
    Node("CL:0000786", "CL:0000542", "blood_immune", "type", "Plasma cell", "浆细胞", "Antibody-secreting B lineage cell."),
    Node("CL:0000623", "CL:0000542", "blood_immune", "type", "Natural killer cell", "NK 细胞", "Innate cytotoxic lymphocyte."),
    Node("CL:0001065", "CL:0000542", "blood_immune", "type", "Innate lymphoid cell", "先天淋巴样细胞", "Innate lymphoid branch."),
    Node("CL:0000766", "CL:0000738", "blood_immune", "group", "Myeloid leukocytes", "髓系白细胞", "Myeloid immune branch."),
    Node("CL:0000576", "CL:0000766", "blood_immune", "type", "Monocyte", "单核细胞", "Circulating mononuclear phagocyte."),
    Node("CL:0000235", "CL:0000766", "blood_immune", "type", "Macrophage", "巨噬细胞", "Tissue phagocyte branch."),
    Node("CL:0000451", "CL:0000766", "blood_immune", "type", "Dendritic cell", "树突状细胞", "Antigen-presenting myeloid cell."),
    Node("CL:0000094", "CL:0000766", "blood_immune", "group", "Granulocytes", "粒细胞", "Granulocyte branch."),
    Node("CL:0000775", "CL:0000094", "blood_immune", "type", "Neutrophil", "中性粒细胞", "Canonical granulocyte."),
    Node("CL:0000771", "CL:0000094", "blood_immune", "type", "Eosinophil", "嗜酸性粒细胞", "Canonical granulocyte."),
    Node("CL:0000767", "CL:0000094", "blood_immune", "type", "Basophil", "嗜碱性粒细胞", "Canonical granulocyte."),
    Node("CL:0000097", "CL:0000766", "blood_immune", "type", "Mast cell", "肥大细胞", "Tissue-resident granulocyte-like immune cell."),
    Node("CL:0000039", "CL:0000000", "germline", "branch", "Germ line cells", "生殖系细胞", "Germ cells and gametes."),
    Node("CL:0000586", "CL:0000039", "germline", "group", "Germ cells", "生殖细胞", "Generic germ cell branch."),
    Node("CL:0000023", "CL:0000586", "germline", "type", "Oocyte", "卵母细胞", "Female germ cell branch."),
    Node("CL:0000019", "CL:0000586", "germline", "type", "Sperm", "精子", "Male gamete."),
    Node("CL:0000017", "CL:0000586", "germline", "type", "Spermatocyte", "精母细胞", "Male germ cell development."),
    Node("CL:0000018", "CL:0000586", "germline", "type", "Spermatid", "精子细胞", "Male germ cell development."),
    Node("CL:0000163", "CL:0000000", "endocrine", "branch", "Endocrine cells", "内分泌细胞", "Hormone-secreting endocrine cells."),
    Node("CL:0008024", "CL:0000163", "endocrine", "group", "Pancreatic endocrine cells", "胰腺内分泌细胞", "Pancreatic islet branch."),
    Node("CL:0000171", "CL:0008024", "endocrine", "type", "Pancreatic alpha cell", "胰腺 α 细胞", "Glucagon-producing pancreatic endocrine cell."),
    Node("CL:0000169", "CL:0008024", "endocrine", "type", "Pancreatic beta cell", "胰腺 β 细胞", "Insulin-producing pancreatic endocrine cell."),
    Node("CL:0000173", "CL:0008024", "endocrine", "type", "Pancreatic delta cell", "胰腺 δ 细胞", "Somatostatin-producing pancreatic endocrine cell."),
    Node("CL:0000147", "CL:0000000", "pigment", "branch", "Pigment cells", "色素细胞", "Pigment-producing or pigment-containing cells."),
    Node("CL:0000148", "CL:0000147", "pigment", "type", "Melanocyte", "黑色素细胞", "Canonical pigment cell."),
]


OFFICIAL_SLIM_ROWS = [
    ("CL:0000000", "", "root", "Cell", "细胞", "Root of the catalog."),
    ("CL:0011115", "CL:0000000", "stem_progenitor", "Precursor cells", "前体细胞", "Broad precursor branch used to place stem and progenitor states."),
    ("CL:0000034", "CL:0011115", "stem_progenitor", "Stem cell", "干细胞", "CL general_cell_types_upper_slim."),
    ("CL:0011026", "CL:0011115", "stem_progenitor", "Progenitor cell", "祖细胞", "Broad CL progenitor term; domain scaffold supplement."),
    ("CL:0002322", "CL:0000034", "stem_progenitor", "Embryonic stem cell", "胚胎干细胞", "CellGuide/HRA-supported stem-cell supplement."),
    ("CL:0000134", "CL:0000034", "stem_progenitor", "Mesenchymal stem cell", "间充质干细胞", "CellGuide/HRA-supported stem-cell supplement."),

    ("CL:0002321", "CL:0000000", "developmental", "Embryonic and developmental cells", "胚胎与发育细胞", "Developmental scaffold supplement."),
    ("CL:0000349", "CL:0002321", "developmental", "Extraembryonic cell", "胚外细胞", "CL general_cell_types_upper_slim."),
    ("CL:0010017", "CL:0002321", "developmental", "Zygote", "合子", "Developmental scaffold supplement."),

    ("CL:0000066", "CL:0000000", "epithelial", "Epithelial cell", "上皮细胞", "CL general_cell_types_upper_slim."),
    ("CL:0000646", "CL:0000066", "epithelial", "Basal cell", "基底细胞", "Broad epithelial scaffold supplement."),
    ("CL:0000312", "CL:0000066", "epithelial", "Keratinocyte", "角质形成细胞", "CellGuide/HRA-supported epithelial supplement."),
    ("CL:0000067", "CL:0000066", "epithelial", "Ciliated epithelial cell", "纤毛上皮细胞", "CellGuide/HRA-supported epithelial supplement."),
    ("CL:1100001", "CL:0000066", "epithelial", "Secretory epithelial cell", "分泌型上皮细胞", "Broad epithelial scaffold supplement."),
    ("CL:0000160", "CL:1100001", "epithelial", "Goblet cell", "杯状细胞", "CellGuide/HRA-supported epithelial supplement."),
    ("CL:0000158", "CL:1100001", "epithelial", "Club cell", "Club 细胞", "CellGuide/HRA-supported epithelial supplement."),
    ("CL:0000584", "CL:0000066", "epithelial", "Enterocyte", "肠吸收细胞", "Broad epithelial scaffold supplement."),
    ("CL:0000182", "CL:0000066", "epithelial", "Hepatocyte", "肝细胞", "Broad epithelial scaffold supplement."),
    ("CL:1000488", "CL:0000066", "epithelial", "Cholangiocyte", "胆管细胞", "Broad epithelial scaffold supplement."),

    ("CL:0000115", "CL:0000000", "endothelial", "Endothelial cell", "内皮细胞", "CL general_cell_types_upper_slim."),
    ("CL:0002139", "CL:0000115", "endothelial", "Vascular endothelial cell", "血管内皮细胞", "Broad endothelial scaffold supplement."),
    ("CL:0000071", "CL:0002139", "endothelial", "Blood vessel endothelial cell", "血管内皮细胞", "CellGuide/HRA-supported endothelial supplement."),
    ("CL:2000008", "CL:0002139", "endothelial", "Microvascular endothelial cell", "微血管内皮细胞", "CellGuide/HRA-supported endothelial supplement."),
    ("CL:0002138", "CL:0000115", "endothelial", "Lymphatic endothelial cell", "淋巴管内皮细胞", "CellGuide/HRA-supported endothelial supplement."),

    ("CL:0002320", "CL:0000000", "stromal_connective", "Connective tissue cell", "结缔组织细胞", "CL general_cell_types_upper_slim."),
    ("CL:0000499", "CL:0002320", "stromal_connective", "Stromal cell", "基质细胞", "CellGuide/HRA-supported stromal scaffold."),
    ("CL:0000057", "CL:0000499", "stromal_connective", "Fibroblast", "成纤维细胞", "CellGuide/HRA-supported stromal supplement."),
    ("CL:0008019", "CL:0002320", "stromal_connective", "Mesenchymal cell", "间充质细胞", "CellGuide/HRA-supported stromal supplement."),
    ("CL:0008034", "CL:0002320", "stromal_connective", "Mural cell", "血管壁细胞", "CellGuide/HRA-supported perivascular supplement."),
    ("CL:0000669", "CL:0008034", "stromal_connective", "Pericyte", "周细胞", "CellGuide/HRA-supported perivascular supplement."),
    ("CL:0000136", "CL:0002320", "stromal_connective", "Adipocyte", "脂肪细胞", "CellGuide/HRA-supported stromal supplement."),
    ("CL:0000138", "CL:0002320", "stromal_connective", "Chondrocyte", "软骨细胞", "CellGuide/HRA-supported stromal supplement."),

    ("CL:0001035", "CL:0000000", "bone", "Bone cell", "骨细胞谱系", "CL general_cell_types_upper_slim."),
    ("CL:0000062", "CL:0001035", "bone", "Osteoblast", "成骨细胞", "CellGuide/HRA-supported bone supplement."),
    ("CL:0000137", "CL:0001035", "bone", "Osteocyte", "骨细胞", "CellGuide/HRA-supported bone supplement."),
    ("CL:0000092", "CL:0001035", "bone", "Osteoclast", "破骨细胞", "CellGuide/HRA-supported bone supplement."),

    ("CL:0000187", "CL:0000000", "muscle", "Muscle cell", "肌肉细胞", "CL general_cell_types_upper_slim."),
    ("CL:0000056", "CL:0000187", "muscle", "Myoblast", "成肌细胞", "CellGuide/HRA-supported myogenic supplement."),
    ("CL:0000192", "CL:0000187", "muscle", "Smooth muscle cell", "平滑肌细胞", "CellGuide/HRA-supported muscle supplement."),
    ("CL:0000359", "CL:0000192", "muscle", "Vascular smooth muscle cell", "血管平滑肌细胞", "CellGuide/HRA-supported muscle supplement."),
    ("CL:0008002", "CL:0000187", "muscle", "Skeletal muscle fiber", "骨骼肌纤维", "CellGuide/HRA-supported muscle supplement."),
    ("CL:0002494", "CL:0000187", "muscle", "Cardiocyte", "心肌细胞谱系", "CellGuide/HRA-supported cardiac supplement."),
    ("CL:0000746", "CL:0002494", "muscle", "Cardiac muscle cell", "心肌细胞", "CellGuide/HRA-supported cardiac supplement."),

    ("CL:0002319", "CL:0000000", "neural", "Neural cell", "神经系统细胞", "CL general_cell_types_upper_slim."),
    ("CL:0011020", "CL:0002319", "neural", "Neural progenitor cell", "神经祖细胞", "CellGuide/HRA-supported neural scaffold."),
    ("CL:0000540", "CL:0002319", "neural", "Neuron", "神经元", "CellGuide/HRA-supported neural scaffold."),
    ("CL:0000101", "CL:0000540", "neural", "Sensory neuron", "感觉神经元", "CellGuide/HRA-supported neuron supplement."),
    ("CL:0000100", "CL:0000540", "neural", "Motor neuron", "运动神经元", "CellGuide/HRA-supported neuron supplement."),
    ("CL:0000099", "CL:0000540", "neural", "Interneuron", "中间神经元", "CellGuide/HRA-supported neuron supplement."),
    ("CL:0000617", "CL:0000540", "neural", "GABAergic neuron", "GABA 能神经元", "CellGuide/HRA-supported neuron supplement."),
    ("CL:0000679", "CL:0000540", "neural", "Glutamatergic neuron", "谷氨酸能神经元", "CellGuide/HRA-supported neuron supplement."),
    ("CL:0000700", "CL:0000540", "neural", "Dopaminergic neuron", "多巴胺能神经元", "CellGuide/HRA-supported neuron supplement."),
    ("CL:0000125", "CL:0002319", "neural", "Glial cell", "胶质细胞", "CellGuide/HRA-supported glial scaffold."),
    ("CL:0000127", "CL:0000125", "neural", "Astrocyte", "星形胶质细胞", "CellGuide/HRA-supported glial supplement."),
    ("CL:0000128", "CL:0000125", "neural", "Oligodendrocyte", "少突胶质细胞", "CellGuide/HRA-supported glial supplement."),
    ("CL:0002453", "CL:0000125", "neural", "Oligodendrocyte precursor cell", "少突胶质前体细胞", "CellGuide/HRA-supported glial supplement."),
    ("CL:0000129", "CL:0000125", "neural", "Microglial cell", "小胶质细胞", "CellGuide/HRA-supported glial supplement."),
    ("CL:0002573", "CL:0000125", "neural", "Schwann cell", "施万细胞", "CellGuide/HRA-supported glial supplement."),
    ("CL:0000065", "CL:0000125", "neural", "Ependymal cell", "室管膜细胞", "CellGuide/HRA-supported glial supplement."),

    ("CL:0000988", "CL:0000000", "blood_immune", "Hematopoietic cell", "造血细胞", "CL general_cell_types_upper_slim."),
    ("CL:0008001", "CL:0000988", "blood_immune", "Hematopoietic precursor cell", "造血前体细胞", "Blood/immune scaffold supplement."),
    ("CL:0000037", "CL:0008001", "blood_immune", "Hematopoietic stem cell", "造血干细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000837", "CL:0008001", "blood_immune", "Hematopoietic multipotent progenitor cell", "造血多能祖细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0002031", "CL:0008001", "blood_immune", "Hematopoietic lineage restricted progenitor cell", "造血谱系限制祖细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0002032", "CL:0008001", "blood_immune", "Hematopoietic oligopotent progenitor cell", "造血寡能祖细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000038", "CL:0008001", "blood_immune", "Erythroid progenitor cell", "红系祖细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000081", "CL:0000988", "blood_immune", "Blood cell", "血细胞", "Blood/immune scaffold supplement."),
    ("CL:0000232", "CL:0000081", "blood_immune", "Erythrocyte", "红细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000547", "CL:0000081", "blood_immune", "Proerythroblast", "原红细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000765", "CL:0000081", "blood_immune", "Erythroblast", "成红细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000558", "CL:0000081", "blood_immune", "Reticulocyte", "网织红细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000556", "CL:0000081", "blood_immune", "Megakaryocyte", "巨核细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000233", "CL:0000081", "blood_immune", "Platelet", "血小板", "CL blood_and_immune_upper_slim."),
    ("CL:0000762", "CL:0000081", "blood_immune", "Nucleated thrombocyte", "有核血小板", "CL blood_and_immune_upper_slim."),
    ("CL:0000738", "CL:0000988", "blood_immune", "Leukocyte", "白细胞", "Blood/immune scaffold supplement."),
    ("CL:0000145", "CL:0000738", "blood_immune", "Professional antigen presenting cell", "专职抗原呈递细胞", "CL blood_and_immune_upper_slim; functional immune category."),
    ("CL:0000842", "CL:0000738", "blood_immune", "Mononuclear leukocyte", "单核白细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0002087", "CL:0000738", "blood_immune", "Nongranular leukocyte", "无颗粒白细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000542", "CL:0000738", "blood_immune", "Lymphocyte", "淋巴细胞", "Blood/immune scaffold supplement."),
    ("CL:0000084", "CL:0000542", "blood_immune", "T cell", "T 细胞", "Blood/immune scaffold supplement."),
    ("CL:0000789", "CL:0000084", "blood_immune", "Alpha-beta T cell", "αβ T 细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000798", "CL:0000084", "blood_immune", "Gamma-delta T cell", "γδ T 细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0002420", "CL:0000084", "blood_immune", "Immature T cell", "未成熟 T 细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000236", "CL:0000542", "blood_immune", "B cell", "B 细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000816", "CL:0000236", "blood_immune", "Immature B cell", "未成熟 B 细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000786", "CL:0000542", "blood_immune", "Plasma cell", "浆细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0001065", "CL:0000542", "blood_immune", "Innate lymphoid cell", "先天淋巴样细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0002679", "CL:0000542", "blood_immune", "Natural helper lymphocyte", "天然辅助淋巴细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0017005", "CL:0000542", "blood_immune", "Lymphoblast", "淋巴母细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0017006", "CL:0017005", "blood_immune", "B-lymphoblast", "B 淋巴母细胞", "CL blood_and_immune_upper_slim."),
    ("CL:4030029", "CL:0000542", "blood_immune", "Blood lymphocyte", "血液淋巴细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000766", "CL:0000738", "blood_immune", "Myeloid leukocyte", "髓系白细胞", "Blood/immune scaffold supplement."),
    ("CL:0000576", "CL:0000766", "blood_immune", "Monocyte", "单核细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000235", "CL:0000766", "blood_immune", "Macrophage", "巨噬细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000451", "CL:0000766", "blood_immune", "Dendritic cell", "树突状细胞", "Blood/immune scaffold supplement."),
    ("CL:0000784", "CL:0000451", "blood_immune", "Plasmacytoid dendritic cell", "浆细胞样树突状细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000990", "CL:0000451", "blood_immune", "Conventional dendritic cell", "经典树突状细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000094", "CL:0000766", "blood_immune", "Granulocyte", "粒细胞", "Blood/immune scaffold supplement."),
    ("CL:0000775", "CL:0000094", "blood_immune", "Neutrophil", "中性粒细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000771", "CL:0000094", "blood_immune", "Eosinophil", "嗜酸性粒细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000767", "CL:0000094", "blood_immune", "Basophil", "嗜碱性粒细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000097", "CL:0000766", "blood_immune", "Mast cell", "肥大细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000647", "CL:0000766", "blood_immune", "Multinucleated giant cell", "多核巨细胞", "CL blood_and_immune_upper_slim."),
    ("CL:0000889", "CL:0000766", "blood_immune", "Myeloid suppressor cell", "髓系抑制细胞", "CL blood_and_immune_upper_slim."),

    ("CL:0000039", "CL:0000000", "germline", "Germ line cell", "生殖系细胞", "CL general_cell_types_upper_slim."),
    ("CL:0000586", "CL:0000039", "germline", "Germ cell", "生殖细胞", "Germline scaffold supplement."),
    ("CL:0000021", "CL:0000586", "germline", "Female germ cell", "雌性生殖细胞", "Germline scaffold supplement."),
    ("CL:0000675", "CL:0000021", "germline", "Female gamete", "雌配子", "Germline scaffold supplement."),
    ("CL:0000023", "CL:0000675", "germline", "Oocyte", "卵母细胞", "Germline scaffold supplement."),
    ("CL:0000024", "CL:0000021", "germline", "Oogonial cell", "卵原细胞", "Germline scaffold supplement."),
    ("CL:0000015", "CL:0000586", "germline", "Male germ cell", "雄性生殖细胞", "Germline scaffold supplement."),
    ("CL:0000408", "CL:0000015", "germline", "Male gamete", "雄配子", "Germline scaffold supplement."),
    ("CL:0000019", "CL:0000408", "germline", "Sperm", "精子", "Germline scaffold supplement."),
    ("CL:0000020", "CL:0000015", "germline", "Spermatogonium", "精原细胞", "Germline scaffold supplement."),
    ("CL:0000017", "CL:0000015", "germline", "Spermatocyte", "精母细胞", "Germline scaffold supplement."),
    ("CL:0000018", "CL:0000015", "germline", "Spermatid", "精子细胞", "Germline scaffold supplement."),

    ("CL:0000163", "CL:0000000", "endocrine", "Endocrine cell", "内分泌细胞", "Broad endocrine scaffold supplement."),
    ("CL:0008024", "CL:0000163", "endocrine", "Pancreatic endocrine cell", "胰腺内分泌细胞", "CellGuide/HRA-supported endocrine supplement."),
    ("CL:0000171", "CL:0008024", "endocrine", "Pancreatic alpha cell", "胰腺 α 细胞", "CellGuide/HRA-supported endocrine supplement."),
    ("CL:0000169", "CL:0008024", "endocrine", "Pancreatic beta cell", "胰腺 β 细胞", "CellGuide/HRA-supported endocrine supplement."),
    ("CL:0000173", "CL:0008024", "endocrine", "Pancreatic delta cell", "胰腺 δ 细胞", "CellGuide/HRA-supported endocrine supplement."),

    ("CL:0000147", "CL:0000000", "pigment", "Pigment cell", "色素细胞", "Pigment scaffold supplement."),
    ("CL:0000148", "CL:0000147", "pigment", "Melanocyte", "黑色素细胞", "CL general_cell_types_upper_slim."),
]


def rows_to_nodes(rows: list[tuple[str, str, str, str, str, str]]) -> list[Node]:
    child_counts: dict[str, int] = defaultdict(int)
    for cl_id, parent_id, *_ in rows:
        if parent_id:
            child_counts[parent_id] += 1
    nodes: list[Node] = []
    for cl_id, parent_id, branch, label, label_zh, note in rows:
        if not parent_id:
            rank = "root"
        elif parent_id == "CL:0000000":
            rank = "branch"
        elif child_counts.get(cl_id, 0):
            rank = "group"
        else:
            rank = "type"
        nodes.append(Node(cl_id, parent_id, branch, rank, label, label_zh, note))
    return nodes


SKELETON = LEGACY_AI_SKELETON


COLORS = {
    "root": "#4a5568",
    "stem_progenitor": "#697b2f",
    "developmental": "#766bb0",
    "epithelial": "#278276",
    "endothelial": "#2776a6",
    "stromal_connective": "#9a6b2f",
    "bone": "#7d6d36",
    "muscle": "#b24d45",
    "neural": "#6754b5",
    "blood_immune": "#b15a25",
    "germline": "#2f7f52",
    "endocrine": "#a45a9c",
    "pigment": "#4d5961",
}


def parse_cl_names() -> dict[str, str]:
    names: dict[str, str] = {}
    current_id = ""
    current_name = ""
    obsolete = False
    for raw in OBO_PATH.read_text(encoding="utf-8").splitlines():
        if raw == "[Term]":
            if current_id.startswith("CL:") and current_name and not obsolete:
                names[current_id] = current_name
            current_id = ""
            current_name = ""
            obsolete = False
            continue
        if raw.startswith("[") and raw != "[Term]":
            if current_id.startswith("CL:") and current_name and not obsolete:
                names[current_id] = current_name
            current_id = ""
            current_name = ""
            obsolete = False
            continue
        if raw.startswith("id: "):
            current_id = raw[4:].strip()
        elif raw.startswith("name: "):
            current_name = raw[6:].strip()
        elif raw.startswith("is_obsolete: true"):
            obsolete = True
    if current_id.startswith("CL:") and current_name and not obsolete:
        names[current_id] = current_name
    return names


def depth_by_id(nodes: list[Node]) -> dict[str, int]:
    by_id = {node.cl_id: node for node in nodes}
    depths: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id in depths:
            return depths[node_id]
        parent_id = by_id[node_id].parent_id
        depths[node_id] = 0 if not parent_id else depth(parent_id) + 1
        return depths[node_id]

    for node in nodes:
        depth(node.cl_id)
    return depths


def validate(nodes: list[Node], cl_names: dict[str, str]) -> None:
    ids = [node.cl_id for node in nodes]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    missing_cl = [node.cl_id for node in nodes if node.cl_id not in cl_names]
    node_ids = set(ids)
    missing_parent = [
        (node.cl_id, node.parent_id)
        for node in nodes
        if node.parent_id and node.parent_id not in node_ids
    ]
    child_counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        if node.parent_id:
            child_counts[node.parent_id] += 1
    empty_groups = [
        node.cl_id
        for node in nodes
        if node.rank == "group" and child_counts.get(node.cl_id, 0) == 0
    ]
    typed_parents = [
        node.cl_id
        for node in nodes
        if node.rank == "type" and child_counts.get(node.cl_id, 0) > 0
    ]
    empty_branches = [
        node.cl_id
        for node in nodes
        if node.rank == "branch" and child_counts.get(node.cl_id, 0) == 0
    ]
    if duplicates or missing_cl or missing_parent or empty_groups or typed_parents or empty_branches:
        raise SystemExit(
            f"Invalid skeleton: duplicates={duplicates}; "
            f"missing_cl={missing_cl}; missing_parent={missing_parent}; "
            f"empty_groups={empty_groups}; typed_parents={typed_parents}; "
            f"empty_branches={empty_branches}"
        )


def rows_for_tsv(nodes: list[Node], cl_names: dict[str, str]) -> list[dict[str, object]]:
    depths = depth_by_id(nodes)
    by_id = {node.cl_id: node for node in nodes}
    rows = []
    for order, node in enumerate(nodes):
        rows.append(
            {
                "order": order + 1,
                "level": depths[node.cl_id],
                "rank": node.rank,
                "branch": node.branch,
                "cl_id": node.cl_id,
                "cl_name": cl_names[node.cl_id],
                "catalog_label": node.label,
                "catalog_label_zh": node.label_zh,
                "parent_id": node.parent_id,
                "parent_label": by_id[node.parent_id].label if node.parent_id else "",
                "note": node.note,
            }
        )
    return rows


def write_tsv(rows: list[dict[str, object]]) -> None:
    fields = [
        "order",
        "level",
        "rank",
        "branch",
        "cl_id",
        "cl_name",
        "catalog_label",
        "catalog_label_zh",
        "parent_id",
        "parent_label",
        "note",
    ]
    with (OUT_DIR / "cell_catalog_skeleton.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(nodes: list[Node], cl_names: dict[str, str]) -> None:
    children: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        children[node.parent_id].append(node)
    lines = [
        "# Cell Catalog Skeleton",
        "",
        "This is a compact review skeleton for the baseline cell catalog.",
        "Every node uses an active Cell Ontology CL ID.",
        "",
    ]

    def add(node: Node, depth: int) -> None:
        lines.append(
            f"{'  ' * depth}- {node.label} / {node.label_zh} "
            f"`{node.cl_id}` ({node.rank}, {node.branch})"
        )
        for child in children.get(node.cl_id, []):
            add(child, depth + 1)

    for root in children[""]:
        add(root, 0)
    (OUT_DIR / "cell_catalog_skeleton.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_html(nodes: list[Node], cl_names: dict[str, str]) -> None:
    depths = depth_by_id(nodes)
    branch_counts = defaultdict(int)
    rank_counts = defaultdict(int)
    data = []
    for order, node in enumerate(nodes):
        branch_counts[node.branch] += 1
        rank_counts[node.rank] += 1
        data.append(
            {
                "id": node.cl_id,
                "parent": node.parent_id,
                "branch": node.branch,
                "rank": node.rank,
                "label": node.label,
                "labelZh": node.label_zh,
                "clName": cl_names[node.cl_id],
                "note": node.note,
                "level": depths[node.cl_id],
                "order": order,
            }
        )
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    colors_json = json.dumps(COLORS, ensure_ascii=False, separators=(",", ":"))
    stats_json = json.dumps(
        {
            "nodes": len(nodes),
            "branches": rank_counts["branch"],
            "groups": rank_counts["group"],
            "types": rank_counts["type"],
            "branchCounts": dict(sorted(branch_counts.items())),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cell Catalog Skeleton</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #647085;
      --line: #d8dee8;
      --hover: #edf2f7;
      --selected: #e5f0ff;
      --shadow: 0 10px 28px rgba(20, 28, 42, 0.08);
      --radius: 8px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 18px 22px 14px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 23px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }}
    .stats div {{
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 10px 12px;
      min-width: 0;
    }}
    .stats b {{
      display: block;
      font-size: 20px;
      line-height: 1.1;
    }}
    .stats span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(240px, 1fr) 190px 130px auto auto auto;
      gap: 8px;
      align-items: center;
      padding: 12px 22px;
      background: #edf1f6;
      border-bottom: 1px solid var(--line);
    }}
    input, select, button {{
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
      min-width: 0;
    }}
    input, select {{
      padding: 0 10px;
    }}
    button {{
      padding: 0 12px;
      cursor: pointer;
      white-space: nowrap;
    }}
    button:hover {{
      background: var(--hover);
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(460px, 1fr) 380px;
      gap: 14px;
      padding: 14px 22px 22px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      min-width: 0;
    }}
    .tree-panel {{
      overflow: hidden;
    }}
    .tree-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }}
    .tree-head h2, .details h2 {{
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .tree-count {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    #tree {{
      height: calc(100vh - 216px);
      min-height: 460px;
      overflow: auto;
      padding: 10px 8px 18px;
    }}
    .row {{
      display: grid;
      grid-template-columns: auto 24px minmax(0, 1fr) auto;
      align-items: center;
      min-height: 36px;
      padding-right: 8px;
      border-radius: 6px;
      cursor: pointer;
    }}
    .row:hover {{
      background: var(--hover);
    }}
    .row.selected {{
      background: var(--selected);
    }}
    .indent {{
      width: calc(var(--depth) * 18px);
      height: 100%;
    }}
    .toggle {{
      width: 22px;
      height: 22px;
      display: grid;
      place-items: center;
      border: 0;
      background: transparent;
      color: var(--muted);
      padding: 0;
      font-size: 14px;
      line-height: 1;
    }}
    .toggle.placeholder {{
      visibility: hidden;
    }}
    .label {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .swatch {{
      width: 9px;
      height: 24px;
      border-radius: 4px;
      flex: 0 0 auto;
      background: var(--branch-color);
    }}
    .name {{
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .zh {{
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 210px;
    }}
    .id {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .badges {{
      display: flex;
      align-items: center;
      gap: 5px;
      margin-left: 8px;
    }}
    .badge {{
      height: 22px;
      min-width: 44px;
      padding: 0 8px;
      border: 1px solid var(--line);
      background: #f8fafc;
      color: var(--muted);
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      white-space: nowrap;
    }}
    .badge.rank-branch {{
      color: #325d78;
      background: #e8f3fa;
      border-color: #c8ddec;
    }}
    .badge.rank-group {{
      color: #5d4c22;
      background: #fff5d8;
      border-color: #e8d59b;
    }}
    .badge.rank-type {{
      color: #365d45;
      background: #eaf6ee;
      border-color: #c8e0cf;
    }}
    mark {{
      background: #fff1a8;
      color: inherit;
      padding: 0 1px;
      border-radius: 2px;
    }}
    .details {{
      padding: 14px;
      position: sticky;
      top: 12px;
      height: fit-content;
    }}
    .details .title {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .details .color {{
      width: 12px;
      height: 30px;
      border-radius: 5px;
      background: var(--branch-color);
      flex: 0 0 auto;
    }}
    .details h2 {{
      font-size: 18px;
      line-height: 1.25;
    }}
    .detail-zh {{
      color: var(--muted);
      margin-top: 2px;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0;
    }}
    .meta-grid div {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfe;
      min-width: 0;
    }}
    .meta-grid b {{
      display: block;
      font-size: 18px;
      line-height: 1.15;
      overflow-wrap: anywhere;
    }}
    .meta-grid span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .field {{
      border-top: 1px solid var(--line);
      padding-top: 11px;
      margin-top: 11px;
    }}
    .field label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #26364f;
      background: #f1f4f8;
      border-radius: 4px;
      padding: 2px 5px;
    }}
    .lineage {{
      color: #344055;
      overflow-wrap: anywhere;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--muted);
      font-size: 12px;
      background: #fbfcfe;
    }}
    .legend i {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--branch-color);
    }}
    .empty {{
      color: var(--muted);
      padding: 22px;
      text-align: center;
    }}
    @media (max-width: 1060px) {{
      .toolbar {{
        grid-template-columns: 1fr 1fr 1fr;
      }}
      main {{
        grid-template-columns: 1fr;
      }}
      .details {{
        position: static;
      }}
      #tree {{
        height: 62vh;
      }}
    }}
    @media (max-width: 680px) {{
      header, .toolbar, main {{
        padding-left: 12px;
        padding-right: 12px;
      }}
      .stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .toolbar {{
        grid-template-columns: 1fr;
      }}
      .badges {{
        display: none;
      }}
      .id {{
        display: none;
      }}
      .label {{
        align-items: flex-start;
        flex-direction: column;
        gap: 2px;
      }}
      .swatch {{
        display: none;
      }}
      .name, .zh {{
        white-space: normal;
        max-width: none;
      }}
      main {{
        padding-top: 12px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Cell Catalog Skeleton</h1>
    <div class="stats" id="stats"></div>
  </header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search name, Chinese label, or CL ID">
    <select id="branchFilter" aria-label="Filter by branch"></select>
    <select id="rankFilter" aria-label="Filter by rank"></select>
    <button id="expandAll" type="button">Expand</button>
    <button id="collapseAll" type="button">Collapse</button>
    <button id="reset" type="button">Reset</button>
  </div>
  <main>
    <section class="panel tree-panel">
      <div class="tree-head">
        <h2>Tree</h2>
        <div class="tree-count" id="treeCount"></div>
      </div>
      <div id="tree"></div>
    </section>
    <aside class="panel details" id="details"></aside>
  </main>
  <script>
    const DATA = {data_json};
    const COLORS = {colors_json};
    const SUMMARY = {stats_json};
    const byId = new Map(DATA.map(node => [node.id, node]));
    const children = new Map();
    DATA.forEach(node => {{
      if (!children.has(node.parent)) children.set(node.parent, []);
      children.get(node.parent).push(node);
    }});
    for (const list of children.values()) list.sort((a, b) => a.order - b.order);

    const root = DATA.find(node => !node.parent) || DATA[0];
    let selectedId = root.id;
    let searchText = "";
    let branchFilter = "all";
    let rankFilter = "all";
    let collapsed = new Set(DATA.filter(node => node.parent && node.level > 1).map(node => node.id));

    function branchColor(branch) {{
      return COLORS[branch] || "#647085";
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function highlight(value) {{
      const safe = escapeHtml(value);
      if (!searchText) return safe;
      const escaped = searchText.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&");
      return safe.replace(new RegExp("(" + escaped + ")", "ig"), "<mark>$1</mark>");
    }}

    function ancestorIds(id) {{
      const ids = [];
      let node = byId.get(id);
      while (node && node.parent) {{
        ids.push(node.parent);
        node = byId.get(node.parent);
      }}
      return ids;
    }}

    function descendantIds(id) {{
      const ids = [];
      const stack = [...(children.get(id) || [])];
      while (stack.length) {{
        const node = stack.pop();
        ids.push(node.id);
        stack.push(...(children.get(node.id) || []));
      }}
      return ids;
    }}

    function nodeMatches(node) {{
      const text = [
        node.label,
        node.labelZh,
        node.clName,
        node.id,
        node.branch,
        node.rank
      ].join(" ").toLowerCase();
      const searchOk = !searchText || text.includes(searchText);
      const branchOk = branchFilter === "all" || node.branch === branchFilter;
      const rankOk = rankFilter === "all" || node.rank === rankFilter;
      return searchOk && branchOk && rankOk;
    }}

    function visibleIds() {{
      const ids = new Set();
      DATA.forEach(node => {{
        if (nodeMatches(node)) {{
          ids.add(node.id);
          ancestorIds(node.id).forEach(parent => ids.add(parent));
        }}
      }});
      if (!searchText && branchFilter === "all" && rankFilter === "all") {{
        DATA.forEach(node => ids.add(node.id));
      }}
      return ids;
    }}

    function renderStats() {{
      const stats = [
        ["Nodes", SUMMARY.nodes],
        ["Branches", SUMMARY.branches],
        ["Groups", SUMMARY.groups],
        ["Types", SUMMARY.types]
      ];
      document.getElementById("stats").innerHTML = stats.map(item =>
        '<div><b>' + escapeHtml(item[1]) + '</b><span>' + escapeHtml(item[0]) + '</span></div>'
      ).join("");
    }}

    function renderFilters() {{
      const branchSelect = document.getElementById("branchFilter");
      const rankSelect = document.getElementById("rankFilter");
      const branches = [...new Set(DATA.map(node => node.branch))];
      const ranks = [...new Set(DATA.map(node => node.rank))];
      branchSelect.innerHTML = '<option value="all">All branches</option>' +
        branches.map(branch => '<option value="' + escapeHtml(branch) + '">' + escapeHtml(branch) + '</option>').join("");
      rankSelect.innerHTML = '<option value="all">All ranks</option>' +
        ranks.map(rank => '<option value="' + escapeHtml(rank) + '">' + escapeHtml(rank) + '</option>').join("");
    }}

    function renderTree() {{
      const visible = visibleIds();
      if (searchText || branchFilter !== "all" || rankFilter !== "all") {{
        visible.forEach(id => ancestorIds(id).forEach(parent => collapsed.delete(parent)));
      }}
      const tree = document.getElementById("tree");
      tree.innerHTML = "";
      let shown = 0;

      function addNode(node, depth) {{
        if (!visible.has(node.id)) return;
        shown += 1;
        const childList = (children.get(node.id) || []).filter(child => visible.has(child.id));
        const hasChildren = childList.length > 0;
        const isCollapsed = collapsed.has(node.id);
        const row = document.createElement("div");
        row.className = "row" + (node.id === selectedId ? " selected" : "");
        row.style.setProperty("--branch-color", branchColor(node.branch));
        row.innerHTML =
          '<div class="indent" style="--depth:' + depth + '"></div>' +
          '<button class="toggle ' + (hasChildren ? "" : "placeholder") + '" type="button" title="' + (isCollapsed ? "Expand" : "Collapse") + '">' + (isCollapsed ? "›" : "⌄") + '</button>' +
          '<div class="label"><span class="swatch"></span><span class="name">' + highlight(node.label) + '</span><span class="zh">' + highlight(node.labelZh) + '</span><span class="id">' + highlight(node.id) + '</span></div>' +
          '<div class="badges"><span class="badge rank-' + escapeHtml(node.rank) + '">' + escapeHtml(node.rank) + '</span><span class="badge">L' + escapeHtml(node.level) + '</span></div>';
        row.addEventListener("click", event => {{
          if (event.target.classList.contains("toggle")) return;
          selectedId = node.id;
          renderTree();
          renderDetails();
        }});
        row.querySelector(".toggle").addEventListener("click", () => {{
          if (collapsed.has(node.id)) collapsed.delete(node.id);
          else collapsed.add(node.id);
          renderTree();
        }});
        tree.appendChild(row);
        if (hasChildren && !isCollapsed) childList.forEach(child => addNode(child, depth + 1));
      }}

      addNode(root, 0);
      if (!shown) tree.innerHTML = '<div class="empty">No matching nodes</div>';
      document.getElementById("treeCount").textContent = shown + " / " + DATA.length + " nodes";
    }}

    function renderDetails() {{
      const node = byId.get(selectedId) || root;
      const parent = node.parent ? byId.get(node.parent) : null;
      const childCount = (children.get(node.id) || []).length;
      const descendants = descendantIds(node.id).length;
      const lineage = [...ancestorIds(node.id)].reverse().map(id => byId.get(id)).filter(Boolean).concat(node);
      const details = document.getElementById("details");
      details.style.setProperty("--branch-color", branchColor(node.branch));
      details.innerHTML =
        '<div class="title"><span class="color"></span><div><h2>' + escapeHtml(node.label) + '</h2><div class="detail-zh">' + escapeHtml(node.labelZh) + '</div></div></div>' +
        '<div class="field"><label>CL ID</label><code>' + escapeHtml(node.id) + '</code></div>' +
        '<div class="meta-grid">' +
          '<div><b>' + escapeHtml(node.rank) + '</b><span>Rank</span></div>' +
          '<div><b>L' + escapeHtml(node.level) + '</b><span>Level</span></div>' +
          '<div><b>' + escapeHtml(childCount) + '</b><span>Children</span></div>' +
          '<div><b>' + escapeHtml(descendants) + '</b><span>Descendants</span></div>' +
        '</div>' +
        '<div class="field"><label>CL name</label>' + escapeHtml(node.clName) + '</div>' +
        '<div class="field"><label>Branch</label>' + escapeHtml(node.branch) + '</div>' +
        '<div class="field"><label>Parent</label>' + (parent ? escapeHtml(parent.label) + ' <code>' + escapeHtml(parent.id) + '</code>' : "None") + '</div>' +
        '<div class="field"><label>Lineage</label><div class="lineage">' + lineage.map(item => escapeHtml(item.label)).join(" / ") + '</div></div>' +
        '<div class="field"><label>Note</label>' + escapeHtml(node.note || "") + '</div>' +
        '<div class="legend">' + renderLegend() + '</div>';
    }}

    function renderLegend() {{
      return Object.entries(SUMMARY.branchCounts).map(entry => {{
        const branch = entry[0];
        const count = entry[1];
        return '<span style="--branch-color:' + branchColor(branch) + '"><i></i>' + escapeHtml(branch) + ' (' + escapeHtml(count) + ')</span>';
      }}).join("");
    }}

    document.getElementById("search").addEventListener("input", event => {{
      searchText = event.target.value.trim().toLowerCase();
      renderTree();
    }});
    document.getElementById("branchFilter").addEventListener("change", event => {{
      branchFilter = event.target.value;
      renderTree();
    }});
    document.getElementById("rankFilter").addEventListener("change", event => {{
      rankFilter = event.target.value;
      renderTree();
    }});
    document.getElementById("expandAll").addEventListener("click", () => {{
      collapsed = new Set();
      renderTree();
    }});
    document.getElementById("collapseAll").addEventListener("click", () => {{
      collapsed = new Set(DATA.filter(node => node.id !== root.id).map(node => node.id));
      renderTree();
    }});
    document.getElementById("reset").addEventListener("click", () => {{
      document.getElementById("search").value = "";
      document.getElementById("branchFilter").value = "all";
      document.getElementById("rankFilter").value = "all";
      searchText = "";
      branchFilter = "all";
      rankFilter = "all";
      selectedId = root.id;
      collapsed = new Set(DATA.filter(node => node.parent && node.level > 1).map(node => node.id));
      renderTree();
      renderDetails();
    }});

    renderStats();
    renderFilters();
    renderTree();
    renderDetails();
  </script>
</body>
</html>
"""
    (OUT_DIR / "cell_catalog_skeleton.html").write_text(html_doc, encoding="utf-8")


def write_readme() -> None:
    (OUT_DIR / "README.md").write_text(
        """# Cell Catalog Skeleton

This directory contains a compact first-pass skeleton for the baseline cell catalog.

Files:

- `cell_catalog_skeleton.tsv`: machine-readable node table.
- `cell_catalog_skeleton.md`: text tree for quick review.
- `cell_catalog_skeleton.html`: collapsible visual tree.

Regenerate:

```bash
python3 scripts/build_cell_catalog_skeleton.py
```

Design intent:

- Use active Cell Ontology CL IDs for every node.
- Keep top-level branches broad enough for a baseline directory.
- Avoid species-specific, marker-specific, and transcriptomic cluster leaves.
- Treat this as a review skeleton before importing external evidence from HPA, Cell Taxonomy, CELLxGENE, HuBMAP, or Azimuth.
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cl_names = parse_cl_names()
    validate(SKELETON, cl_names)
    rows = rows_for_tsv(SKELETON, cl_names)
    write_tsv(rows)
    write_markdown(SKELETON, cl_names)
    write_html(SKELETON, cl_names)
    write_readme()
    print(f"Wrote {len(SKELETON)} skeleton nodes to {OUT_DIR}")


if __name__ == "__main__":
    main()
