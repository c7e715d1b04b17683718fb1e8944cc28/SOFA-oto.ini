import sys

sys.path.append("src/SOFA")
sys.path.append("src/SOFA/modules")
import glob
import pathlib
import tqdm
import re
from SOFA.modules.g2p.base_g2p import DataFrameDataset
import pandas as pd
import warnings
import pyopenjtalk
import SOFA.infer
import SOFA.modules.g2p
import SOFA.modules.AP_detector
import torch
from SOFA.train import LitForcedAlignmentTask
import lightning as pl
import utaupy

VERSION = "0.0.1"
HIRAGANA_REGEX = re.compile(r"([あ-ん][ぁぃぅぇぉゃゅょ]|[あ-ん])")


class PyOpenJTalkG2P:
    def __call__(self, text: str):
        ph_seq, word_seq, ph_idx_to_word_idx = self._g2p(text)

        # The first and last phonemes should be `SP`,
        # and there should not be more than two consecutive `SP`s at any position.
        assert ph_seq[0] == "SP" and ph_seq[-1] == "SP"
        assert all(
            ph_seq[i] != "SP" or ph_seq[i + 1] != "SP" for i in range(len(ph_seq) - 1)
        )
        return ph_seq, word_seq, ph_idx_to_word_idx

    def _g2p(self, input_text: str):
        word_seq_raw = input_text.strip().split(" ")
        word_seq = []
        word_seq_idx = 0
        ph_seq = ["SP"]
        ph_idx_to_word_idx = [-1]
        for word in word_seq_raw:
            ph_raw = pyopenjtalk.g2p(word)
            if not ph_raw:
                warnings.warn(f"Word {word} is not in the dictionary. Ignored.")
                continue
            word_seq.append(word)
            phones = ph_raw.split(" ")
            for i, ph in enumerate(phones):
                if (i == 0 or i == len(phones) - 1) and ph == "SP":
                    warnings.warn(
                        f"The first or last phoneme of word {word} is SP, which is not allowed. "
                        "Please check your dictionary."
                    )
                    continue
                ph_seq.append(ph)
                ph_idx_to_word_idx.append(word_seq_idx)
            if ph_seq[-1] != "SP":
                ph_seq.append("SP")
                ph_idx_to_word_idx.append(-1)
            word_seq_idx += 1

        return ph_seq, word_seq, ph_idx_to_word_idx

    def get_dataset(self, wav_path: pathlib.Path):
        dataset = []
        try:
            if wav_path.with_suffix(".txt").exists():
                with open(wav_path.with_suffix(".txt"), "r", encoding="utf-8") as f:
                    lab_text = f.read().strip()
                ph_seq, word_seq, ph_idx_to_word_idx = self(lab_text)
                dataset.append((wav_path, ph_seq, word_seq, ph_idx_to_word_idx))
        except Exception as e:
            e.args = (f" Error when processing {wav_path}: {e} ",)
            raise e
        dataset = pd.DataFrame(
            dataset, columns=["wav_path", "ph_seq", "word_seq", "ph_idx_to_word_idx"]
        )
        return DataFrameDataset(dataset)


def main():
    print(f"SOFA oto.ini {VERSION} - Use SOFA to generate oto.ini.")
    print()

    voicebank_folder_path = input("Enter the path to the voicebank folder: ")
    voicebank_wav_files = glob.glob(voicebank_folder_path + "/*.wav")
    if len(voicebank_wav_files) == 0:
        print("No wav files found in the specified folder.")
        input("Press Enter to exit.")
        return
    generate_vcv_oto_ini_flag = input("Generate VCV oto.ini? (y/n): ") == "y"
    generate_cvvc_oto_ini_flag = input("Generate CVVC oto.ini? (y/n): ") == "y"
    suffix = input("Enter the suffix to add to the after all entries in the oto.ini: ")

    print()
    print("Phase 1: Generating text files...")
    print()

    with tqdm.tqdm(total=len(voicebank_wav_files)) as pbar:
        for wav_file in voicebank_wav_files:
            file_name = pathlib.Path(wav_file).stem
            words = file_name[1:]
            graphemes = HIRAGANA_REGEX.findall(words)
            with open(
                voicebank_folder_path + "/" + file_name + ".txt", "w", encoding="utf-8"
            ) as f:
                f.write(" ".join(graphemes))
            pbar.update(1)

    print()
    print("Phase 1: Done.")
    print()
    print("Phase 2: Generating label files...")
    print()

    AP_detector_class = SOFA.modules.AP_detector.LoudnessSpectralcentroidAPDetector
    get_AP = AP_detector_class()

    g2p_class = PyOpenJTalkG2P
    grapheme_to_phoneme = g2p_class()

    torch.set_grad_enabled(False)

    model = LitForcedAlignmentTask.load_from_checkpoint(
        "src/cktp/japanese-v2.0-45000.ckpt"
    )
    model.set_inference_mode("force")

    trainer = pl.Trainer(logger=False)

    for wav_file in voicebank_wav_files:
        print()
        file_name = pathlib.Path(wav_file).stem
        print(file_name)

        dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(wav_file))

        predictions = trainer.predict(
            model, dataloaders=dataset, return_predictions=True
        )

        predictions = get_AP.process(predictions)
        predictions = SOFA.infer.post_processing(predictions)

        for (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
        ) in predictions:
            label = ""
            for ph, (start, end) in zip(ph_seq, ph_intervals):
                start_time = int(float(start) * 10000000)
                end_time = int(float(end) * 10000000)
                label += f"{start_time} {end_time} {ph}\n"
            with open(
                voicebank_folder_path + "/" + file_name + ".lab", "w", encoding="utf-8"
            ) as f:
                f.write(label)

    print()
    print("Phase 2: Done.")
    print()
    print("Phase 3: Generating oto.ini file...")
    print()

    # TODO: Generate VCV and CVVC oto.ini from .lab
    if generate_vcv_oto_ini_flag:
        # [[Label("k"), Label("a")], [Label("a")]]
        for wav_file in voicebank_wav_files:
            file_name = pathlib.Path(wav_file).stem
            label = utaupy.label.load(voicebank_folder_path + "/" + file_name + ".lab")

    if generate_cvvc_oto_ini_flag:
        pass

    # Convert Easily
    time_order_ratio = 10 ** (-4)
    otoini = utaupy.otoini.OtoIni()
    with tqdm.tqdm(total=len(voicebank_wav_files)) as pbar:
        for wav_file in voicebank_wav_files:
            file_name = pathlib.Path(wav_file).stem
            label = utaupy.label.load(voicebank_folder_path + "/" + file_name + ".lab")
            for phoneme in label:
                oto = utaupy.otoini.Oto()
                oto.filename = file_name + ".wav"
                oto.alias = phoneme.symbol
                oto.offset = phoneme.start * time_order_ratio
                oto.overlap = 0.0
                oto.preutterance = 0.0
                oto.consonant = (phoneme.end - phoneme.start) * time_order_ratio
                oto.cutoff = -(phoneme.end - phoneme.start) * time_order_ratio
                otoini.append(oto)
            pbar.update(1)
    otoini.write(voicebank_folder_path + "/oto-SOFA.ini")

    print()
    print("Phase 3: Done.")
    print()
    input("Press Enter to exit.")


if __name__ == "__main__":
    main()
