import math
import numpy as np
import os
import wave


def nextpow2(n):
    return np.ceil(np.log2(np.abs(n))).astype("long")


def berouti(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 4 - SNR * 3 / 20
    else:
        if SNR < -5.0:
            a = 5
        if SNR > 20:
            a = 1
    return a


def berouti1(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 3 - SNR * 2 / 20
    else:
        if SNR < -5.0:
            a = 4
        if SNR > 20:
            a = 1
    return a


def find_index(x_list):
    index_list = []
    for i in range(len(x_list)):
        if x_list[i] < 0:
            index_list.append(i)
    return index_list


def denoise(audio_file, input_path, output_path):
    file_name = os.path.basename(audio_file).split(".")[0]
    input_file_path = os.path.join(input_path, audio_file)
    output_file_path = os.path.join(output_path, file_name)

    with wave.open(input_file_path, 'r') as f:
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)

    x = np.frombuffer(str_data, dtype=np.short)
    len_ = 20 * framerate // 1000
    len1 = len_ * 50 // 100
    # The percentage of the frame that the window overlaps
    len2 = len_ - len1
    Thres = 3
    Expnt = 2.0
    beta = 0.002
    G = 0.9
    win = np.hamming(len_)
    # normalization gain for overlap+add with 50% overlap
    winGain = len2 / sum(win)

    # Noise magnitude calculations - assuming that the first 5 frames is noise/silence
    nFFT = 2 * 2 ** (nextpow2(len_))
    noise_mean = np.zeros(nFFT)

    j = 0
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * x[j: j + len_], nFFT))
        j = j + len_
    noise_mu = noise_mean / 5

    # --- allocate memory and initialize various variables
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    # Start Processing
    for n in range(0, Nframes):
        # Windowing
        insign = win * x[k - 1: k + len_ - 1]
        # compute fourier transform of a frame
        spec = np.fft.fft(insign, nFFT)
        # compute the magnitude
        sig = abs(spec)

        # save the noisy phase information
        theta = np.angle(spec)
        SNRseg = 10 * np.log10(
            np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2
        )

        if Expnt == 1.0:
            alpha = berouti1(SNRseg)
        else:
            alpha = berouti(SNRseg)

        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt

        diffw = sub_speech - beta * noise_mu ** Expnt

        z = find_index(diffw)
        if len(z) > 0:
            sub_speech[z] = beta * noise_mu[z] ** Expnt
            # implement a simple VAD detector
        if SNRseg < Thres:  # Update noise spectrum
            noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt
            noise_mu = noise_temp ** (1 / Expnt)
        sub_speech[nFFT // 2 + 1: nFFT] = np.flipud(sub_speech[1: nFFT // 2])
        x_phase = (sub_speech ** (1 / Expnt)) * (
                np.array([math.cos(x) for x in theta])
                + img * (np.array([math.sin(x) for x in theta]))
        )
        # take the IFFT

        xi = np.fft.ifft(x_phase).real
        # Overlap and add
        xfinal[k - 1: k + len2 - 1] = x_old + xi[0:len1]
        x_old = xi[0 + len1: len_]
        k = k + len2

    with wave.open(output_file_path + "-denoised.wav", "wb") as wf:
        wf.setparams(params)
        wave_data = (winGain * xfinal).astype(np.short)
        wf.writeframes(wave_data.tobytes())
        print(f"writing to {output_file_path}-denoised.wav")
