import glob
import pathlib
import tqdm
import pyopenjtalk
import sys

sys.path.append("src/SOFA")
import SOFA.infer
import SOFA.modules.g2p
import SOFA.modules.AP_detector
import torch
from SOFA.train import LitForcedAlignmentTask
import lightning as pl
import utaupy

VERSION = "0.0.1"


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
            phonemes = (
                pyopenjtalk.g2p(words)
                .replace("A", "a")
                .replace("I", "i")
                .replace("U", "u")
                .replace("E", "e")
                .replace("O", "o")
            )
            with open(
                voicebank_folder_path + "/" + file_name + ".txt", "w", encoding="utf-8"
            ) as f:
                f.write(phonemes)
            pbar.update(1)

    print()
    print("Phase 1: Done.")
    print()
    print("Phase 2: Generating label files...")
    print()

    AP_detector_class = SOFA.modules.AP_detector.LoudnessSpectralcentroidAPDetector
    get_AP = AP_detector_class()

    g2p_class = SOFA.modules.g2p.DictionaryG2P
    grapheme_to_phoneme = g2p_class(dictionary="src/dictionary/japanese-dictionary.txt")
    grapheme_to_phoneme.set_in_format("txt")

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

        dataset = grapheme_to_phoneme.get_dataset([pathlib.Path(wav_file)])

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
        pass
    
    if generate_cvvc_oto_ini_flag:
        pass
    
    # Convert Easily
    # time_order_ratio = 10 ** (-4)
    # otoini = utaupy.otoini.OtoIni()
    # with tqdm.tqdm(total=len(voicebank_wav_files)) as pbar:
    #     for wav_file in voicebank_wav_files:
    #         file_name = pathlib.Path(wav_file).stem
    #         label = utaupy.label.load(voicebank_folder_path + "/" + file_name + ".lab")
    #         for phoneme in label:
    #             oto = utaupy.otoini.Oto()
    #             oto.filename = file_name + ".wav"
    #             oto.alias = phoneme.symbol
    #             oto.offset = phoneme.start * time_order_ratio
    #             oto.overlap = 0.0
    #             oto.preutterance = 0.0
    #             oto.consonant = (phoneme.end - phoneme.start) * time_order_ratio
    #             oto.cutoff = -(phoneme.end - phoneme.start) * time_order_ratio
    #             otoini.append(oto)
    #         pbar.update(1)
    # otoini.write(voicebank_folder_path + "/oto.ini")

    print()
    print("Phase 3: Done.")
    print()
    input("Press Enter to exit.")


if __name__ == "__main__":
    main()
