import torch
import torchaudio
import sr_indie_rnn.modules as srirnn
import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str, default='../../audio_datasets/dist_fx_192k/44k/')

if __name__ == '__main__':
    args = parser.parse_args()

    model_dir = 'Proteus_Tone_Packs'
    model_files = [
            '6505Plus_Red_DirectOut.json',
            'MesaIICplus_Drive8_5EQoff.json',
            'BlackstarHT40_AmpClean.json',		'MesaMiniRec_HighGain_DirectOut.json',
            'BlackstarHT40_AmpHighGain.json',	'PlexiBreed_JTM_pedal.json',
            'BossMT2_PedalHighGain.json',   	'PrincetonAmp_Clean.json',
            'DumbleKit_HighG_DirectOut.json',	'ProcoRatPedal_HighGain.json',
            'DumbleKit_LowG_DirectOut.json',	'RockmanAcoustic_Pedal.json',
            'GoatPedal_HighGain.json',      	'RockmanXPR_HighGain.json',
            'LittleBigMuff_HighGainPedal.json', 'Splawn_OD_FractalFM3_HighGain.json',
            'MatchlessSC30_Ch1_DirectOut.json',	'XComp_Pedal.json']

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        base_dir = args.data_dir
        input_files = ['train/ht1-input.wav',
                        'val/ht1-input.wav',
                        'test/ht1-input.wav']
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        model = srirnn.get_AudioRNN_from_json(model_path)
        model.eval()

        with torch.no_grad():
            for file in input_files:
                in_sig, sample_rate = torchaudio.load(os.path.join(base_dir, file))
                out_sig, _ = model(in_sig)

                save_name_input = os.path.join(base_dir, os.path.dirname(file),'-'.join((model_name, 'input.wav')))
                torchaudio.save(save_name_input, in_sig, sample_rate)
                print(save_name_input)

                save_name_target = os.path.join(base_dir, os.path.dirname(file),'-'.join((model_name, 'target.wav')))
                torchaudio.save(save_name_target, out_sig.detach(), sample_rate)
                print(save_name_target)


