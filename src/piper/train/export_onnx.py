#!/usr/bin/env python3

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__name__)
OPSET_VERSION = 15


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output-file", required=True, help="Path to output file (.onnx)"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)

    # pylint: disable=no-value-for-parameter
    model = VitsModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model_g = model.model_g

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales, sid=None, lid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
            lid=lid,
        )[0].unsqueeze(1)

        return audio

    model_g.forward = infer_forward  # type: ignore[method-assign,assignment]

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers
    num_languages = model_g.n_languages

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    if num_speakers > 1:
        sid = torch.LongTensor([0])

    lid: Optional[torch.Tensor] = None
    if num_languages > 1:
        # One language id per phoneme id
        lid = torch.randint(
            low=0, high=num_languages, size=sequences.shape, dtype=torch.long
        )

    # noise, length, noise_w
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid, lid)

    # Inputs that are None are not part of the graph, so name only the ones
    # that are (otherwise "lid" would be labeled "sid" on single-speaker models).
    input_names = ["input", "input_lengths", "scales"]
    dynamic_axes = {
        "input": {0: "batch_size", 1: "phonemes"},
        "input_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 2: "time"},
    }
    if sid is not None:
        input_names.append("sid")

    if lid is not None:
        input_names.append("lid")
        dynamic_axes["lid"] = {0: "batch_size", 1: "phonemes"}

    # PyTorch >= 2.9 defaults to the dynamo exporter, which needs onnxscript and
    # ignores dynamic_axes. Keep the TorchScript exporter this script targets.
    export_kwargs: Dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    # Export
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=output_path,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        **export_kwargs,
    )
    _LOGGER.info("Exported model to %s", output_path)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
