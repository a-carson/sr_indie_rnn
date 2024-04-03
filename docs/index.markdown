#### Welcome to the accompanying web-page for our DAFx24 submission <sup>1</sup>.


### <b>Abstract</b>
In recent years, machine learning approaches to modelling guitar amplifiers and effects pedals have been widely investigated and have become standard practice in some consumer products. In particular, recurrent neural networks (RNNs) are a popular choice for modelling non-linear devices such as vacuum tube amplifiers and distortion circuitry. One limitation of such models is that they are trained on audio at a specific sample rate and therefore give unreliable results when operating at another rate. Here, we investigate several methods of modifying RNN structures to make them approximately sample rate independent, with a focus on oversampling. In the case of integer oversampling, we demonstrate that a previously proposed delay-based approach provides high fidelity sample rate conversion whilst additionally reducing aliasing. For non-integer sample rate adjustment, we propose two novel methods and show that one of these, based on cubic Lagrange interpolation of a delay-line, provides a significant improvement over existing methods. To our knowledge, this work provides the first in-depth study into this problem.
##### <sup>1</sup> Alistair Carson, Alec Wright, Jatin Chowdhury, Vesa Välimäki and Stefan Bilbao "Sample rate independent recurrent neural networks for audio effects processing". Submitted to *Proceedings of the 27th International Conference on Digital Audio Effects (DAFx24)*, Guildford, UK, Sept. 2024. Subject to peer review.

### <b>Audio Examples</b>

Peavey 6505Plus tube amp -- high gain 

Input signal: bass guitar example
<table style="text-align: center">
  <thead>
    <tr>
      <th colspan="3" style="border-left-style: hidden; border-top-style: hidden; visibility:  hidden"></th>
      <th colspan="3" style="text-align: center">Delay-based methods</th>
    </tr>
    <tr>
      <th>Inference sample rate</th>
      <th>Original RNN (trained at 44.1kHz) </th>
      <th>STN  </th>
      <th>LIDL  </th>
      <th>APDL (ours) </th>
      <th>CIDL (ours) </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>44.1kHz</td>
      <td >
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_44100_naive_riff1.wav" type="audio/wav">
        </audio></td>
      <td>
        -</td>
      <td>
        -</td>
      <td>
        -</td>
      <td>
        - </td>
    </tr>
    <tr>
      <td>48kHz</td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_48000_naive_riff1.wav" type="audio/wav">        </audio></td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_48000_stn_riff1.wav" type="audio/wav">        </audio></td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_48000_lidl_riff1.wav" type="audio/wav">        </audio></td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_48000_apdl_riff1.wav" type="audio/wav">        </audio></td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_48000_lagrange_riff1.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td>88.2kHz</td>
      <td >
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_88200_naive_riff1.wav" type="audio/wav">        </audio></td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_88200_stn_riff1.wav" type="audio/wav">        </audio></td>
      <td colspan="3" style="text-align: center">
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_88200_lidl_riff1.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td>96kHz</td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_96000_naive_riff1.wav" type="audio/wav">        </audio></td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_96000_stn_riff1.wav" type="audio/wav">        </audio></td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_96000_lidl_riff1.wav" type="audio/wav">        </audio></td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_96000_apdl_riff1.wav" type="audio/wav">        </audio></td>
      <td>
        <audio controls>
          <source src="audio/6505Plus_Red_DirectOut_96000_lagrange_riff1.wav" type="audio/wav">        </audio></td>
    </tr>
  </tbody>
</table>
