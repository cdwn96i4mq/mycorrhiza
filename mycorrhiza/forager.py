import os
from typing import Tuple

import gin
import numpy as np
import plac
import torch
from torch import Tensor

from mycorrhiza.cultivation.grower import Grower
from mycorrhiza.field.loupe import get_logger
from mycorrhiza.foray.exclusion import exclusion_score, flag_zones
from mycorrhiza.foray.leiden import leiden_clusters
from mycorrhiza.foray.programs import discover_programs, merge_programs
from mycorrhiza.keys.domains import adjusted_rand_index, normalised_mutual_information
from mycorrhiza.network import ImmunoFounder
from mycorrhiza.substrate.corpus import CrossModalBatch, synthetic_corpus
from mycorrhiza.substrate.lattice import neighbour_index

_LOG = get_logger("mycorrhiza.forager")


def _configure(conf: str) -> None:
    gin.parse_config_file(conf)


def _spatial_slice(batch: CrossModalBatch, embeddings: Tensor) -> Tuple[Tensor, Tensor]:
    spot = (batch.modality == 1).nonzero(as_tuple=False).flatten()
    return spot, embeddings[spot]


class Forager:
    commands = ("cultivate", "key", "forage", "export")

    @plac.annotations(
        conf=("gin configuration file", "option", "c", str),
        epochs=("training passes over the corpus", "option", "e", int),
        out=("checkpoint name", "option", "o", str),
    )
    def cultivate(self, conf="conf/symbiont.gin", epochs=1, out="latest.pt"):
        # type: (str, int, str) -> None
        _configure(conf)
        batch = synthetic_corpus()
        model = ImmunoFounder(batch.expr.shape[1])
        grower = Grower(model)
        history = grower.fit([batch], epochs)
        path = grower.save(out)
        _LOG.info(
            "cultivated %d steps, final loss %.4f, checkpoint %s", len(history), history[-1], path
        )

    @plac.annotations(
        conf=("gin configuration file", "option", "c", str),
        resolution=("leiden resolution", "option", "r", float),
    )
    def key(self, conf="conf/symbiont.gin", resolution=1.0):
        # type: (str, float) -> None
        _configure(conf)
        batch = synthetic_corpus()
        model = ImmunoFounder(batch.expr.shape[1])
        model.eval()
        with torch.no_grad():
            embeddings = model.embed(batch)
        spot, spatial = _spatial_slice(batch, embeddings)
        clusters = leiden_clusters(spatial, resolution=resolution)
        truth = batch.label[spot].cpu().numpy().astype(np.int_)
        ari = adjusted_rand_index(truth, clusters)
        nmi = normalised_mutual_information(truth, clusters)
        _LOG.info("spatial domains: ARI %.3f NMI %.3f over %d spots", ari, nmi, spot.numel())

    @plac.annotations(
        conf=("gin configuration file", "option", "c", str),
        threshold=("immune-exclusion cutoff", "option", "t", float),
    )
    def forage(self, conf="conf/symbiont.gin", threshold=0.5):
        # type: (str, float) -> None
        _configure(conf)
        batch = synthetic_corpus()
        model = ImmunoFounder(batch.expr.shape[1])
        model.eval()
        with torch.no_grad():
            embeddings = model.embed(batch)
        spot, spatial = _spatial_slice(batch, embeddings)
        clusters = leiden_clusters(spatial)
        neighbours = neighbour_index(batch.coords[spot], 6).cpu().numpy().astype(np.int_)
        immune = batch.immune[spot].cpu().numpy().astype(np.float64)
        scores = exclusion_score(clusters, immune, neighbours)
        flagged = flag_zones(scores, threshold)
        expr = batch.expr[spot].cpu().numpy().astype(np.float64)
        programs = discover_programs(expr, clusters, flagged)
        merged = merge_programs(programs)
        _LOG.info(
            "flagged %d exclusion zones, %d merged evasion programs", flagged.size, len(merged)
        )

    @plac.annotations(
        conf=("gin configuration file", "option", "c", str),
        out=("onnx output path", "option", "o", str),
    )
    def export(self, conf="conf/symbiont.gin", out="reconstructor.onnx"):
        # type: (str, str) -> None
        _configure(conf)
        batch = synthetic_corpus()
        model = ImmunoFounder(batch.expr.shape[1])
        model.eval()
        dim = model.reconstructor.head.in_features
        dummy = torch.zeros((1, dim))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        torch.onnx.export(
            model.reconstructor,
            dummy,
            out,
            input_names=["embedding"],
            output_names=["expression"],
            opset_version=13,
            dynamic_axes={"embedding": {0: "tokens"}, "expression": {0: "tokens"}},
        )
        _LOG.info("exported reconstructor head to %s", out)


def main(argv=None):
    # type: (object) -> None
    plac.call(Forager(), argv)


if __name__ == "__main__":
    main()
