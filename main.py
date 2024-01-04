import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from numba import jit
from multiprocessing import Pool
import os
import time


def getImage(index, grayscale=False, scale=0.5):
    if grayscale:
        grayscale = 0
    else:
        grayscale = 1
    gt = cv2.imread('data/Image' + str(index) + '.png', grayscale)
    gt = cv2.resize(gt, (0, 0), fx=scale, fy=scale)
    return gt


def getNoisyImage(index, grayscale=False, scale=0.5):
    if grayscale:
        grayscale = 0
    else:
        grayscale = 1
    gt = cv2.imread('data/s&p/0.05noisy/Image' + str(index) + '_0.05_noisy.png', grayscale)
    gt = cv2.resize(gt, (0, 0), fx=scale, fy=scale)

    return gt


def log(index, gtImg, noisy, gfiltered, nlmfiltered, params, time_taken_nl):
    f = open('OUTPUT/LOGS/' + str(index) + '-LOG.csv', 'a')

    f.write('Salt and Pepper Noise\n')
    f.write('Params: ' + str(params) + '\n')
    f.write('NOISY,GAUSSIAN FILTER on NOISE,NLM FILTER on NOISE\n')
    f.write(str(peak_signal_noise_ratio(gtImg, noisy)))
    f.write(',')
    f.write(str(peak_signal_noise_ratio(gtImg, gfiltered)))
    f.write(',')
    f.write(str(peak_signal_noise_ratio(gtImg, nlmfiltered)))
    f.write('\n\n')
    f.write(str(time_taken_nl))
    f.write('\n\n')


@jit(nopython=True)
def nonLocalMeans(noisy, params=tuple()):
    bigWindowSize, smallWindowSize, h = params
    padwidth = bigWindowSize // 2
    image = noisy.copy()

    paddedImage = np.zeros((image.shape[0] + bigWindowSize, image.shape[1] + bigWindowSize))
    paddedImage = paddedImage.astype(np.uint8)
    paddedImage[padwidth:padwidth + image.shape[0], padwidth:padwidth + image.shape[1]] = image
    paddedImage[padwidth:padwidth + image.shape[0], 0:padwidth] = np.fliplr(image[:, 0:padwidth])
    paddedImage[padwidth:padwidth + image.shape[0],
    image.shape[1] + padwidth:image.shape[1] + 2 * padwidth] = np.fliplr(
        image[:, image.shape[1] - padwidth:image.shape[1]])
    paddedImage[0:padwidth, :] = np.flipud(paddedImage[padwidth:2 * padwidth, :])
    paddedImage[padwidth + image.shape[0]:2 * padwidth + image.shape[0], :] = np.flipud(
        paddedImage[paddedImage.shape[0] - 2 * padwidth:paddedImage.shape[0] - padwidth, :])

    outputImage = paddedImage.copy()
    smallhalfwidth = smallWindowSize // 2

    diff_weight = [1.0, 0.36787944117144233, 0.1353352832366127, 0.049787068367863944, 0.01831563888873418,
                   0.006737946999085467, 0.0024787521766663585, 0.0009118819655545162, 0.00033546262790251185,
                   0.00012340980408667956, 4.5399929762484854e-05, 1.670170079024566e-05, 6.14421235332821e-06,
                   2.2603294069810542e-06, 8.315287191035679e-07, 3.059023205018258e-07, 1.1253517471925912e-07,
                   4.139937718785167e-08, 1.522997974471263e-08, 5.602796437537268e-09, 2.061153622438558e-09,
                   7.582560427911907e-10, 2.7894680928689246e-10, 1.026187963170189e-10, 3.775134544279098e-11,
                   1.3887943864964021e-11, 5.109089028063325e-12, 1.8795288165390832e-12, 6.914400106940203e-13,
                   2.543665647376923e-13, 9.357622968840175e-14, 3.442477108469977e-14, 1.2664165549094176e-14,
                   4.658886145103398e-15, 1.713908431542013e-15, 6.305116760146989e-16, 2.3195228302435696e-16,
                   8.533047625744066e-17, 3.1391327920480296e-17, 1.1548224173015786e-17, 4.248354255291589e-18,
                   1.5628821893349888e-18, 5.74952226429356e-19, 2.1151310375910805e-19, 7.781132241133797e-20,
                   2.8625185805493937e-20, 1.0530617357553812e-20, 3.873997628687187e-21, 1.4251640827409352e-21,
                   5.242885663363464e-22, 1.9287498479639178e-22, 7.095474162284704e-23, 2.6102790696677047e-23,
                   9.602680054508676e-24, 3.532628572200807e-24, 1.2995814250075031e-24, 4.780892883885469e-25,
                   1.7587922024243116e-25, 6.47023492564546e-26, 2.3802664086944007e-26, 8.75651076269652e-27,
                   3.221340285992516e-27, 1.185064864233981e-27, 4.359610000063081e-28, 1.603810890548638e-28,
                   5.900090541597061e-29, 2.1705220113036395e-29, 7.984904245686979e-30, 2.9374821117108028e-30,
                   1.0806392777072785e-30, 3.975449735908647e-31, 1.462486227251231e-31, 5.380186160021138e-32,
                   1.9792598779469045e-32, 7.281290178321643e-33, 2.6786369618080778e-33, 9.854154686111257e-34,
                   3.6251409191435593e-34, 1.3336148155022614e-34, 4.906094730649281e-35, 1.8048513878454153e-35,
                   6.639677199580735e-36, 2.4426007377405277e-36, 8.985825944049381e-37, 3.3057006267607343e-37,
                   1.2160992992528256e-37, 4.4737793061811207e-38, 1.6458114310822737e-38, 6.054601895401186e-39,
                   2.2273635617957438e-39, 8.194012623990515e-40, 3.0144087850653746e-40, 1.1089390193121365e-40,
                   4.0795586671775603e-41, 1.5007857627073948e-41, 5.5210822770285325e-42, 2.031092662734811e-42,
                   7.47197233734299e-43, 2.7487850079102147e-43, 1.0112214926104486e-43, 3.720075976020836e-44,
                   1.368539471173853e-44, 5.0345753587649823e-45, 1.8521167695179754e-45, 6.813556821545298e-46,
                   2.506567475899953e-46, 9.221146422925876e-47, 3.392270193026015e-47, 1.2479464629129513e-47,
                   4.590938473882946e-48, 1.6889118802245324e-48, 6.213159586848109e-49, 2.2856936767186716e-49,
                   8.408597124803643e-50, 3.093350011308561e-50, 1.1379798735078682e-50, 4.1863939993042314e-51,
                   1.5400882849875202e-51, 5.665668176358939e-52, 2.0842828425817514e-52, 7.667648073722e-53,
                   2.820770088460135e-53, 1.0377033238158346e-53, 3.817497188671175e-54, 1.4043787324419038e-54,
                   5.166420632837861e-55, 1.9006199352650016e-55, 6.991989996645917e-56, 2.572209372642415e-56,
                   9.462629465836378e-57, 3.4811068399043105e-57, 1.2806276389220833e-57, 4.7111658015535965e-58,
                   1.733141042341547e-58, 6.375869581278994e-59, 2.3455513385429143e-59, 8.628801156620959e-60,
                   3.1743585474772134e-60, 1.1677812485237086e-60, 4.2960271311739114e-61, 1.580420060273613e-61,
                   5.814040485895939e-62, 2.138865964899539e-62, 7.868448159078602e-63, 2.8946403116483003e-63,
                   1.0648786602415064e-63, 3.917469664450395e-64, 1.4411565509640892e-64, 5.301718666092324e-65,
                   1.9503933001302485e-65, 7.175095973164411e-66, 2.6395702969591894e-66, 9.710436457780846e-67,
                   3.572269937619218e-67, 1.314164668364901e-67, 4.834541638053336e-68, 1.7785284761271306e-68,
                   6.542840619051457e-69, 2.4069765506104637e-69, 8.854771883513433e-70, 3.257488532207521e-70,
                   1.1983630608508849e-70, 4.408531331463226e-71, 1.6218080426054863e-71, 5.96629836401057e-72,
                   2.194878508014299e-72, 8.074506789675094e-73, 2.970445045520691e-73, 1.0927656633766312e-73,
                   4.020060215743355e-74, 1.4788975056432133e-74, 5.440559879258653e-75, 2.001470128041443e-75,
                   7.362997122252211e-76, 2.7086952666810816e-76, 9.964733010103672e-77, 3.665820411179563e-77,
                   1.3485799642996046e-77, 4.961148436415422e-78, 1.8251045143570802e-78, 6.714184288211594e-79,
                   2.470010363869359e-79, 9.086660323479307e-80, 3.3427955219162848e-80, 1.2297457485529627e-80,
                   4.52398178760621e-81, 1.664279891894355e-81, 6.122543565829638e-82, 2.252357905545217e-82,
                   8.285961676100547e-83, 3.0482349509718567e-83, 1.1213829703227856e-83, 4.125337404615185e-84,
                   1.5176268190534823e-84, 5.583037061001886e-85, 2.0538845540408258e-85, 7.555819019711961e-86,
                   2.779630478564191e-86, 1.0225689071173033e-86, 3.761820781096061e-87, 1.3838965267367376e-87,
                   5.09107080895011e-88, 1.8729002841608093e-88, 6.89001509906914e-89, 2.534694904308355e-89,
                   9.324621449370601e-90, 3.4303365279297016e-90, 1.2619502849247644e-90, 4.642455656042647e-91,
                   1.7078639924081707e-91, 6.282880511239462e-92, 2.3113425714217192e-92, 8.502954135303866e-93,
                   3.1280620156019908e-93, 1.1507497062492758e-93, 4.23337158863185e-94, 1.5573703742969461e-94,
                   5.729245429933205e-95, 2.107671607097867e-95, 7.753690529920792e-96, 2.852423339163565e-96,
                   1.0493479039958717e-96, 3.860335205164256e-97, 1.4201379580102718e-97, 5.22439558379172e-98,
                   1.921947727823849e-98, 7.070450560725609e-99, 2.601073401110048e-99, 9.568814292462674e-100,
                   3.5201700545844787e-100, 1.2949981925089835e-100, 4.764032113782328e-101, 1.7525894717410477e-101,
                   6.4474163546704995e-102, 2.371871925555801e-102, 8.72562918503701e-103, 3.209979588460643e-103,
                   1.1808854971746377e-103, 4.344234967880666e-104, 1.598154732301378e-104, 5.8792826982452694e-105,
                   2.1628672335193993e-105, 7.9567438919514e-106, 2.927122496515368e-106, 1.0768281882584307e-106,
                   3.961429521341682e-107, 1.4573284785512322e-107, 5.361211862926555e-108, 1.9722796241351285e-108,
                   7.255611259606534e-109, 2.6691902155412764e-109, 9.819402048736065e-110, 3.6123561383267394e-110,
                   1.3289115574798703e-110, 4.888792411319657e-111, 1.7984862202794635e-111]

    for imageX in range(padwidth, padwidth + image.shape[1]):
        for imageY in range(padwidth, padwidth + image.shape[0]):

            bWinX = imageX - padwidth
            bWinY = imageY - padwidth

            compNbhd = paddedImage[imageY - smallhalfwidth:imageY + smallhalfwidth + 1,
                       imageX - smallhalfwidth:imageX + smallhalfwidth + 1]

            pixelColor = 0
            totalWeight = 0

            for sWinX in range(bWinX, bWinX + bigWindowSize - smallWindowSize, 1):
                for sWinY in range(bWinY, bWinY + bigWindowSize - smallWindowSize, 1):
                    weight_sum = 0

                    smallNbhd = paddedImage[sWinY:sWinY + smallWindowSize + 1, sWinX:sWinX + smallWindowSize + 1]
                    diff = np.abs(np.subtract(smallNbhd, compNbhd))

                    for i in range(7):
                        for j in range(7):
                            weight_sum += diff_weight[diff[i, j]]
                    totalWeight += weight_sum
                    pixelColor += weight_sum * paddedImage[sWinY + smallhalfwidth, sWinX + smallhalfwidth]

            pixelColor /= totalWeight
            outputImage[imageY, imageX] = pixelColor

    return outputImage[padwidth:padwidth + image.shape[0], padwidth:padwidth + image.shape[1]]


def denoise(index):
    print('DENOISING IMAGE', index)

    f = open('output/logs/' + str(index) + '-LOG.csv', 'w')
    f.close()

    scale = 2
    gtImg = getImage(index, grayscale=True, scale=scale)
    gtNoisyImg = getNoisyImage(index, grayscale=True, scale=scale)

    saltNoised = gtNoisyImg

    kernelSize = 3
    kernel = (kernelSize, kernelSize)

    saltParams = {
        'bigWindow': 14,
        'smallWindow': 2,
        'h': 1,
        'scale': scale,
    }

    start_time_nl = time.time()
    nlmFilteredSalted = nonLocalMeans(saltNoised,
                                      params=(saltParams['bigWindow'], saltParams['smallWindow'], saltParams['h']))
    time_taken_nl = time.time() - start_time_nl
    print(time_taken_nl)

    gFilteredSalted = cv2.GaussianBlur(saltNoised, kernel, 0)

    log(index, gtImg, saltNoised, gFilteredSalted, nlmFilteredSalted, saltParams, time_taken_nl)

    cv2.imwrite('OUTPUT/NOISED/' + str(index) + '-SPNOISE.png', saltNoised)
    cv2.imwrite('OUTPUT/NLMFILTER/' + str(index) + '-NLM-Salted.png', nlmFilteredSalted)
    cv2.imwrite('OUTPUT/GFILTER/' + str(index) + '-GF-Salted.png', gFilteredSalted)
    cv2.imwrite('OUTPUT/GT/' + str(index) + '-GT.png', gtImg)
    print("--------COMPLETED IMAGE", index, '-----------')


if __name__ == '__main__':
    denoise(1)
