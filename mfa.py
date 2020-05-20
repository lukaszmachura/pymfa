# Copyright 2019 Lukasz Machura, lukasz.machura@us.edu.pl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt


def fmfssc(data, polyorder=[1], scales=[], qorder=list(range(-15,16)),
           focus=None,
           with_zero_q=True, shuffle=False, raw_data=False,
           ndiff=None, walking_window=None,
           quiet=True, plots=False, eps=0.0001, maxiter=10000):
    """
    ***
    * based on
    * 1. "Introduction to multifractal detrended fluctuation analysis in
    *     Matlab",
    *  	 E. A. F. Ihlen, Front. Physio. 3, 141 (2012)
    * 2. "Multifractal detrended fluctuation analysis of nonstationary time
    *     series",
    *	 J. W. Kantelhardt, S. A. Zschiegner, E. Koscielny-Bundec, S. Havlin,
    *	 A. Bunde and H. E. Stanley, Phys. A 316, 87 (2002)
    * 3. "Multifractal formalism by enforcing the universal behavior of scaling
    *     functions"
    *    P. Mukli, Z. Nagy and A. Eke, Phys. A 417, 150 (2015).
    * 4. "Multifractal Properties of BK Channels’ Currents in Human
    *     Glioblastoma Cells"
    *    A. Wawrzkiewicz-Jałowiecka, P. Trybek, B. Dworakowska and L. Machura
    *    J. Phys. Chem. B 124, 12 2382-2391 (2020)
    ***

    A function that calculates a series of parameters that characterize a time
    series. The analysis is based on the assumption of multiractical properties
    of the series. As a result, the Hurst exponent of the order 'q' is
    calculated (q-order Hurst exponent).

    Parameters
    ----------
      data : list or numpy.array
          tested time series, 1D
      polyorder : list or tuple
          a sequence containing degrees of polynomial fited to the data in a
          given time window; the default degree is linear i.e. [1]
      scales : list or tuple
          list containing the sizes of windows used for the dfa. Best - powers
          of 2; if the list is empty (by default) the scales are calculated
      qorder : list or tuple
          list of values for which the RMS and the corresponding Hurst exponent
          of order q is calculated; default qorder = list(range(-15,16));
      focus : None or number (int/float) or 'Hurst'
          None : (default) takes the scale function value calculated fo the
                 total length (L) of the time series (S_L) as a focus point
                 (L, S_L)
          number : user can set the value as a focus point, thus focus point
                 will be set as (L, number)
          'Hurst' : an extension of the linear fit of S(q=2) i.e. value of
                 original Hurst value H = H_{q=2} is taken as a focus point,
                 thus the focus poitn will be (L, S(q=2))
      with_zero_q : bool
            whether to calculate the Hurst exponent for q = 0
            True : (default) calculates F(q=0) if it is present in qorder param
            False : opposite
      shuffle : bool
            whether to perform analysis on original or shuffled data
            True : shuffles the data before the analysis
            False : (default) opposite
      ndiff : None or int
            whether to differentiate the data (typical method to get data
            stationarity)
            None : (default) keep original data
            int :  degree of differentiation (1, 2, 3...)
      raw_data : bool
            True : analysis pefrormed on original data
            False : (default) analysis performed on integrated data
                    (cumulatively summed values, typical DFA method)
      walking_window : None or int
            whether to overlap windows in DFA, better not use, only in calse of
            very short data length
            None : (default) separated segments
            int : overlaped segments, overlap of INT number of points (1,2,3...)
      quiet : bool
            True : (default) keep the calculations quiet
            False: be (somewhat) verbose
      plots : bool,
            Uses matplotlib.
            True : plot MSEq(q) for testing purposes, produces 'mseq.png'
            False : (default) doesn't plot anything
      eps : float
            accuracy of iterative fit, stops if eps reached
            default eps=0.0001
      maxiter : int
            iterative fit stops if maxiter reached regardless the accuracy eps
            (helps to keep the iteration finite)
            default maxiter=10000

    Returns
    -------
      dictionary
          {'data': data, 'BW': X, 'm': polyorder, 'scales': scales,
           'q': list(qorder), 'Fq': Fq, 'RMS': RMS, 'qRMS': qRMS,
           'Hq': Hq, 'bq': bq, 'bHq': bHq, 'tq': tq, 'hq': hq, 'Dq': Dq,
           'SL': dSL}

        holds params provided
        m : list
            list of polynomial odres used in DFA, dict['m']
        scales : list
            list of scales used in the DFA analysis, dict['scales']
        q : list
            list of powers for qRMS, Fq i Hq without 0, dict['q']
        data : list
            original data, dict['data']

        holds output of the multifractal analysis
        'BW': list
            keeps the integrated (cumsum) data, dict['BW']
        RMS : dictionary
            holds the local trend matched by the polynomial 'm'-th order for
            the given segment (cut to a multiple of the window length),
            dict['RMS'][m][s]
        qRMS : dictionary
            as above but of order 'q'; useful to locate large (high positive
            valules of q) and small fluctuations (high negative values of q)
	        for a certain polynomial order 'm' and for segments of length 's',
            dict['qRMS'][m][s][q]
        Fq : dictionary
            fluctuation function of order 'q', kept for certain polynomial
            order 'm' and for segments of length 's', dict['Fq'][m][s][q]
        Hq : dictionary
            keeps linear fit (y = ax + b) of log(Fq) vs log(scales)
            characteristics, for certain 'q'-order and 'm'-order, q-order Hurst
            exponent is first element of tuple kept, i.e
            Hq = a = dict['Hq'][m][q][0], b = dict['Hq'][m][q][1]
        bq : dictionary
            keeps the set of the free parameters (intercepts) 'b' of the linear
            fit (y = ax + b) of log(Fq) vs log(scales) characteristics based on
            the focus point based algorithm, for certain 'q' and 'm'
        bHq : dictionary
            keeps the set of the Hurst parameters (intercepts) 'a' of the linear
            fit (y = ax + b) of log(Fq) vs log(scales) characteristics based on
            the focus point based algorithm, for certain 'q' and 'm'
        SL: dictionary
            keeps the focus points for all polynomials orders 'm'
        tq : dictionary
            q-order mass exponent, dict['tq'][m][q]
        hq : dictionary
            q-order singularity strenght, dict['hq'][m][q]
        Dq : dictionary
            q-order singularity dimension, dict['Dq'][m][q]
	"""

    import numpy, warnings


    def model(x, a, x1, y1):
        """linear model for FMF"""
        return a * (x - x1) + y1


    def fit_to_focus(x, y, model, focus_point):
        """fit to model with the focus point, see Mukli or LM"""
        from scipy.optimize import curve_fit

        x1, y1 = focus_point
        init_vals = [0] + focus_point
        v, c = curve_fit(model,
                         x + [x1], y + [y1],
                         p0=init_vals)
        return v


    def gradient_descent(Fq, q, scales, Hq, SL,
                         mm=None, b=None, maks=maxiter, eps=eps, learning_rate=0.01,
                         verbose=False,
                         printmse=False):
        """
        iterative fit of set of linear representations to the q-order
        fluctuation functions
        """

        if b == None:
            b = 0
        else:
            b = b

        if mm == None:
            mm = [0 for i in range(len(Hq))]
        else:
            mm = Hq[:]

        i = 0
        bufMSE, MSE = 2 * eps, 0
        while (i < maks) and (abs(bufMSE - MSE) > eps):
            i += 1
            bufMSE = MSE

            if (not i % (maks // 10)) and verbose:
                print(i, mm[0], Hq[0], b, np.log2(SL))

            delta_b = 0
            delta_m = []
            MSEq = []
            for iq in range(len(q)):
                mdeltatmp = 0
                bufmseq = 0
                for iscale in range(len(scales)):
                    x = np.log2(scales[iscale]) - np.log2(len(data))
                    y = np.log2(Fq[iq][iscale])

                    guess = mm[iq] * x + b
                    mdeltatmp += (-guess + y) * x
                    delta_b += -guess + y

                    bufmseq += (-guess + y) ** 2

                MSEq.append(bufmseq / len(scales))
                delta_m.append(mdeltatmp/ len(scales))
            delta_b /= (len(q) * len(scales))


            # new m
            for iq in range(len(q)):
                mm[iq] = mm[iq] + delta_m[iq] * learning_rate

            # new b
            b = b + delta_b * learning_rate

            MSE = np.mean(MSEq)

        if printmse:
            import matplotlib.pyplot as plt
            plt.plot(q, MSEq, 'o-', label="%d iters" % i)
            plt.xlabel('q')
            plt.ylabel('MSE(q)')
            plt.legend()
            plt.savefig('mseq.png')

        #if i >= maks:
        #    print("""Maximum number of iterations (%) exceeded (%).
        #          Might not converged. Try to set maxiter to higher number.""" % (maks, i))

        return mm


    # numpy.array
    X = numpy.array(data)

    # roznicowanie danych (do stacjonarnosci)
    if ndiff != None:
        X = numpy.diff(X, n=ndiff)

    # charakterystyki dla pomieszanych danych
    if shuffle:
        numpy.random.shuffle(X)

    # raw or integrated data
    if raw_data:
        X = X - numpy.mean(X)
    else:
        X = numpy.cumsum(X - numpy.mean(X))

    # spr czy jest 0 w potegach i usunac na zyczenie...
    if not with_zero_q and 0 in qorder: qorder.remove(0)

    #automatic scales
    if scales == []:
        # opcja 1: max segment = 1/10th of data
        scales = [2 ** i for i in range(4, int(numpy.floor(numpy.log2(len(data) / 10))) + 2)]

        # todo: like Ihlen
        # opcja 2: max segment = sqrt(len/50)
        # max_scale = int(np.sqrt(len(data) / 50))

    q_eps = 0.5 * abs(qorder[1] - qorder[0])

    RMS_m = []
    qRMS_m = []
    Fq_m = []

    # order of polynomial in DFA
    for m in polyorder:
        RMS_scale = []
        qRMS_scale = []
        Fq_scale = []
        for ns in scales:
            if walking_window == None:
                step = ns
                segments = int(numpy.floor(len(X) / ns))
            else:
                assert(0 < walking_window < ns)
                step = walking_window
                segments = int(numpy.floor(len(X)/step)) - ns - 1

            if not quiet:
                print("len(X) %d, step %d, segments %d, step*segments %d"%(len(X), step, segments, step * segments))

            RMS_dla_skali = []
            for v in range(segments):
                idx_start = v * step
                idx_stop  = idx_start + ns

                C = numpy.polyfit(range(idx_start, idx_stop), X[idx_start:idx_stop], m)
                fit = numpy.poly1d(C)
                _b = numpy.sqrt(numpy.mean((X[idx_start:idx_stop] - fit(range(idx_start,idx_stop))) ** 2))
                # ***
                RMS_dla_skali.append(_b)

            qRMS_dla_skali = []
            for q in qorder:
                if -0.001 < q < 0.001:  # zero
                    qRMS_dla_skali.append(numpy.array(RMS_dla_skali) ** 2)
                else:
                    qRMS_dla_skali.append(numpy.array(RMS_dla_skali) ** (float(q)))
                    # Note: there is no q/2 here, as there is squered root
                    # above already
            qRMS_dict_dla_skali = dict(zip(qorder, qRMS_dla_skali))

            Fq_dla_skali = []
            for q in qorder:
                if -0.001 < q < 0.001:
                    Fq_dla_skali.append(numpy.exp(0.5 * numpy.mean(numpy.log(qRMS_dict_dla_skali[q]))))
                else:
                    Fq_dla_skali.append(numpy.mean(qRMS_dict_dla_skali[q]) ** (1.0 / float(q)))
            Fq_dict_dla_skali = dict(zip(qorder, Fq_dla_skali))

            RMS_scale.append(RMS_dla_skali)
            qRMS_scale.append(qRMS_dict_dla_skali)
            Fq_scale.append(Fq_dict_dla_skali)

        RMS_dict_scale = dict(zip(scales, RMS_scale))
        qRMS_dict_scale = dict(zip(scales, qRMS_scale))
        Fq_dict_scale = dict(zip(scales, Fq_scale))

        Fq_m.append(Fq_dict_scale)
        RMS_m.append(RMS_dict_scale)
        qRMS_m.append(qRMS_dict_scale)

    Fq = dict(zip(polyorder, Fq_m))
    qRMS = dict(zip(polyorder, qRMS_m))
    RMS = dict(zip(polyorder, RMS_m))


    Hq_m = []
    bq_m = []
    bHq_m = []
    tq_m = []
    hq_m = []
    Dq_m = []
    SL_m = []
    for m in polyorder:

        ###
        ### REAL DATA PART
        # scale parameter or focus point
        lX = len(X)
        C = numpy.polyfit(range(0, lX), X, m)
        fit = numpy.poly1d(C)

        _Fq = [sorted([Fq[m][i][j] for i in scales]) for j in qorder]

        Fq_qorder = dict(zip(qorder, _Fq))
        Hq_qorder = []
        tq_qorder = []

        # Hq from real data
        b = []
        for q in qorder:
            C = numpy.polyfit(numpy.log2(numpy.array(scales)),
                              numpy.log2(numpy.array(Fq_qorder[q])),
                              1)
            Hq_qorder.append(C[0])
            b.append(C[1])  # do y = ax + b

            if 2 - q_eps < q < 2 + q_eps:
                global_Hurst_a, global_Hurst_b = C

        _Hq = dict(zip(qorder, Hq_qorder))
        _bq = dict(zip(qorder, b))
        Hq_m.append(_Hq)
        bq_m.append(_bq)
        ###
        ############


        if focus == "Hurst":
            max_scale = lX
            SL = global_Hurst_a * max_scale + global_Hurst_b
            if not quiet:
                print("Hurst based focus point: {}".format(SL))

        elif isinstance(focus, (int, float)):
            SL = focus
            if not quiet:
                print("Fixed value for focus point: {}".format(SL))

        else:
            SL = numpy.sqrt(numpy.mean((X - fit(range(0, lX))) ** 2))

        SL_m.append(SL)


        best_Hq = gradient_descent(_Fq, qorder[:], scales[:],
                         Hq_qorder[:], SL,
                         # initial values
                         mm=Hq_qorder[:], b=np.log2(SL),
                         #maks=5000,
                         learning_rate=0.01,
                         verbose=not quiet,
                         printmse=plots)
        bHq_m.append(dict(zip(qorder, best_Hq)))

        tq_qorder = best_Hq * numpy.array(qorder) - 1.0
        tq_m.append(dict(zip(qorder, tq_qorder)))
        hq_qorder = numpy.diff(tq_qorder) / numpy.diff(qorder)
        hq_m.append(dict(zip(qorder[:-1], hq_qorder.tolist())))
        Dq_m.append(dict(zip(qorder[:-1], (numpy.array(qorder[:-1]) * hq_qorder) - numpy.array(tq_qorder[:-1]))))

    bHq = dict(zip(polyorder, bHq_m)) # q-order Hurst exponent with focus point
    Hq = dict(zip(polyorder, Hq_m)) # q-order Hurst exponent
    bq = dict(zip(polyorder, bq_m)) # q-order Hurst exponen
    tq = dict(zip(polyorder, tq_m)) # q-order mass exponent
    hq = dict(zip(polyorder, hq_m)) # q-order singularity exponent (Hoelder exponent)
    Dq = dict(zip(polyorder, Dq_m)) # q-order singularity dimension
    dSL = dict(zip(polyorder, SL_m)) # focus point

    return {
        'data': data,
        'BW': X,
        'm': polyorder,
        'scales': scales,
        'q': list(qorder),
        'Fq': Fq,
        'RMS': RMS,
        'qRMS': qRMS,
        'Hq': Hq,
        'bq': bq,
        'bHq': bHq,
        'tq': tq,
        'hq': hq,
        'Dq': Dq,
        'SL': dSL
        }


def dataplot_qRMS(dicto, m=1, skala=16, q=1, *arg, **kwarg):
    dx = range(len(dicto['data']))
    dy = []
    for i in dicto['qRMS'][m][skala][q]:
        _qrms = [i for j in range(skala)]
        dy += _qrms
    return zip(dx,dy)


def get_spectrum(dicto, m=None, full=False, with_q_zero=True):
    """get_spectrum(dictionary, m=1, full=False)"""

    if m == None:
        m == dicto['m'][0]

    _q = dicto['q'][:-1]
    _dx = [dicto['hq'][m][q] for q in _q]
    _sdx = sorted(_dx)
    _dxy = dict([(dicto['hq'][m][q],dicto['Dq'][m][q]) for q in _q])
    _zd  = zip(_sdx,[_dxy[i] for i in _sdx])

    if full:
        deltax = _q[1] - _q[0]
        tenpercent = len(_q)//10 + 1

        #rare events: q -> infinity
        qrange = _q[-tenpercent:]
        rare_events = [(dicto['hq'][m][q],dicto['Dq'][m][q]) for q in qrange]

        #smooth part : q -> infinity
        qrange = _q[:tenpercent]
        smooth_part = [(dicto['hq'][m][q],dicto['Dq'][m][q]) for q in qrange]

        hurst = [(dicto['hq'][m][q],dicto['Dq'][m][q]) for q in _q if 2.0-deltax < q < 2.0+deltax][0]
        spectrum_maximum = [(dicto['hq'][m][q],dicto['Dq'][m][q]) for q in _q if -deltax < q < deltax][0]
        half_width = abs(spectrum_maximum[0] - hurst[0])

        ret = {'spectrum':list(_zd),
               'rare':list(rare_events),
               'smooth':smooth_part,
               'max':spectrum_maximum,
               'hurst':hurst,
               'width':half_width}
    else:
        ret = {'spectrum':list(_zd)}

    return ret


def emdmfdfa(data, emd='emd', scales=[], qorder=range(-10,11), with_zero_q=True, shuffle=False, raw_data=False, ndiff=None, walking_window=None, quiet=True):
    """
    ***
    * based on
    * 1. "Introduction to multifractal detrended fluctuation analysis in Matlab",
    *  	E. A. F. Ihlen, Front. Physio. 3, 141 (2012)
    * 2. "Multifractal detrended fluctuation analysis of nonstationary time series",
    *	J. W. Kantelhardt, S. A. Zschiegner, E. Koscielny-Bundec, S. Havlin,
    *	A. Bunde and H. E. Stanley, Phys. A 316, 87 (2002)
    * 3. "Modified detrended fluctuation analysis based on empirical mode decomposition
    * for the characterization of anti-persistent processes."
    * Qian XY, Gu GF, Zhou WX Physica A 390, 4388 (2011)
    ***

    Funkcja obliczajaca szereg parametrow charakteryzujacych szereg czasowy.
    Analiza oparta jest na zalozeniu multifraktalnych wlasnosci szeregu. W wyniku
    obliczany jest wykladnik Hursta rzedu 'q' (q-order Hurst exponent).

    INPUT:
      data    - dane w postaci listy lub numpy.array zawierajace badany szereg czasowy
      	        (1D, zakladamy uporzadkowanie w czasie);
      emd     - typ emd: emd, eemd, ceemdan
      scales  - lista zawierajaca wielkosci okien branych w analizie do obliczania
      		wykladnika Hursta. Powinny zawierac potegi liczby 2;
		jezeli lista jest pusta (domyslnie) skale obliczane sa automatycznie;
      qorder  - lista wartosci dla ktorych obliczany jest RMS rzedu q oraz
      		odpowiadajacy mu wykladnik Hursta rzedu q;
		domyslnie qorder=range(-5,6,2);
      with_zero_q - wartosc decydujaca czy obliczc wykladnik Hursta (i cala reszte)
                    dla q = 0;
		    True - (domyslnie) oblicza F(q=0) jezeli takowa zawarta jest w liscie qorder,
		    False - pomija (lub usuwa) z listy qorder wartosc q=0;
      shuffle - bazujemy na danych oryginalnych, czy przetasowanych losowo;
      		ma to na celu sprawdzenie, czy multifraktalne charakterystyki
		pochodza z korelacji dalekozasiegowych, czy z szerokiego
		rozkladu gestosci prawdopodobienstwa (jezeli multifraktalnosc
		bierze sie z korelacji, to zniknie):
		dla pomieszanych danych;
      		True - miesza losowo dane przed obliczaniem charakterystyk,
		False - (domyslnie) pracuje na oryginalnych danych;
      ndiff - roznicowanie danych, obliczanie przyrostow oryginalnych danych,
	      typowa technika osiagania stacjonarnosci z danych niestacjonarnych:
	      None - (domyslnie) pracuje na oryginalnych danych
	      1,2,3,... - stopien roznicowania
      raw_data - definiuje dane ktore poddajemy analizie
                 True - oblicza charakterystyki dla danych oryginalnych (po odjeciu sredniej)
                 False - (domyslnie) oblicza charakterystyki dla danych scalkowanych (cumsum)
      walking_window - definiuje nalozenie sie segmentow podczas obliczania Fq na siebie
                       None - (domyslnie) segmenty rozdzielne
		       1,2,3,... - nakladajace sie segmenty, jest to krok z jakim segment ma
		       podrozowac po danych

    OUTPUT:

      Slownik
      {'Fq':Fq, 'RMS':RMS, 'qRMS':qRMS, 'm':emd,
      	'scales':scales, 'q':qorder, 'data':data, 'Hq':Hq,
	'tq':tq, 'hq':hq, 'Dq':Dq}

      dane wejsciowe:
      m      - l

      ista rzedow wielomianow uzytych do dopasowania polyfit-em: dict['m']
      scales - lista wielkosci okien (skal) uzytych w analizie, skala
      	       16 <= s <= max(len(data)), zawsze potega liczby 2: dict['scales']
      q      - lista wykladnikow uzytych do okreslenia rzedu qRMS, Fq i Hq; nie
      	       zawiera liczby 0: dict['q']
      data   - lista trzymajaca dane: dict['data']

      dane z analizy:
      RMS    - slownik; trzyma lokalny trend dopasowany wielomianem 'm'-tego rzedu dla
      	       danej skali 's' dla calosci danych (obcietych do wielokrotnosci dlugosci
	       okna):  dict['RMS'][m][s]
      qRMS   - slownik; zwraca lokalny trend rzedu 'q' - funkcje probkojaca gdzie w
               szeregu wystepuja duze (duze dodatnie q) i male (ujemne q) fluktuacje
	       dla dopasowania wielomianem rzedu 'm', okien o skali (wielkosci) 's':
	       dict['qRMS'][m][s][q]
      Fq     - slownik; funkcja fluktuacji rzedu 'q' dla danego rzedu dopasowania
	       wielomianu 'm' i skali 's' (wielkosci okien): dict['Fq'][m][s][q]
      Hq     - slownik; zwraca liniowe (y=ax+b) dopasowanie do log(Fq) vs log(scales)
      	       dla rzedow q i m (jak wyzej): wykladnik Hursta rzedu q jest pierwszym
	       elementem dopasowania: Hq = a = dict['Hq'][m][q][0], b = dict['Hq'][m][q][1]
      tq     - slownik; wykladnik masowy rzedu q (q-order mass exponent): dict['tq'][m][q]
      hq     - slownik; natezenie osobliwosci rzedu q lub wykladnik Hoersta (q-order
      	       singularity strenght): dict['hq'][m][q]
      Dq     - slownik; wymiar osobliwosci rzedu q (q-order singularity dimension):
               dict['Dq'][m][q]

    EXAMPLES:
      zakladamy, ze szereg mamy w liscie 'dane' (moze ty byc lista pythonowa lub
      ndarray numpy)

      sage: dict_mfdfa1 = mfdfa1(dane)
        zwroci nam slownik wielkosci dla m = 1, wielkosci okien scales=[16,32,64,128] oraz
        wykladnikow dopasowania RMS rzedu q=[-3,-2,-1,1,2,3]

      sage: max_range = numpy.floor(numpy.log2(dane/10)) + 2
      sage: _sca = [2**i for i in range(4,max_range)]
      sage: _q = range(-9,0)+range(1,10)
      sage: dict_emdmfdfa = emdmfdfa(dane, emd='emd', scales=_sca, qorder=_q)
        zwroci nam slownik wielkosci dla m = [1,2,3] (liniowe, kwadratowe i szescianowe
	dopasowanie), wielkosci okien scales=[16,32,64,128,...,MAX2] (od 16 az do okna
	o szerokosci mieszczacej sie tylko raz w obrebie danych), oraz wykladnikow
	dopasowania RMS rzedu q=[-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9]

	"""

    import numpy, warnings, pyeemd

    # numpy.array
    X = numpy.array(data)

    # roznicowanie danych (do stacjonarnosci)
    if ndiff != None:
      X = numpy.diff(X, n=ndiff)

    # charakterystyki dla pomieszanych danych
    if shuffle:
      numpy.random.shuffle(X)

    X = X - numpy.mean(X)
    # raw or integrated data
    if not raw_data:
      X = numpy.cumsum(X)

    # spr czy jest 0 w potegach i usunac na zyczenie...
    if not with_zero_q and 0 in qorder: qorder.remove(0)

    #automatic scales
    if scales == []:
      scales = [2**i for i in range(4,int(numpy.floor(numpy.log2(len(data)/10))) + 2)]

    RMS_m = []
    qRMS_m = []
    Fq_m = []
    for m in [emd]:
        """rzad wielomianu do dopasowania"""

        RMS_scale = []
        qRMS_scale = []
        Fq_scale = []
        for ns in scales:

	    if walking_window == None:
	      step = ns
              segments = int(numpy.floor(len(X)/ns))
	    else:
	      assert(0 < walking_window < ns)
	      step = walking_window
	      segments = int(numpy.floor(len(X)/step)) - ns - 1

	    if not quiet:
	      print "len(X) %d, step %d, segments %d, step*segments %d"%(len(X), step, segments, step*segments)

            RMS_dla_skali = []
            for v in range(segments):
		idx_start = v * step
		idx_stop  = idx_start + ns

		# EMD - based detrending
		# see: Quian XY, et al. Physica A 390 4388 (2011)
		if emd == 'ceemdan':
		  imfs = pyeemd.ceemdan(X[idx_start:idx_stop])
		elif emd == 'eemd':
		  imfs = pyeemd.eemd(X[idx_start:idx_stop])
		else:
		  imfs = pyeemd.emd(X[idx_start:idx_stop])
		# r = imfs[-1]
		r = X[idx_start:idx_stop] - reduce(numpy.add, imfs[:-1])
		_b = numpy.sqrt(numpy.mean((X[idx_start:idx_stop] - r) ** 2))
  		# ***
                RMS_dla_skali.append(_b)

            qRMS_dla_skali = []
            for q in qorder:
	      if -0.001 < q < 0.001:
		qRMS_dla_skali.append(numpy.array(RMS_dla_skali) ** 2)
	      else:
                qRMS_dla_skali.append(numpy.array(RMS_dla_skali) ** (float(q)))
		# Tu lekkie wyjasnienie na przyszlosc: tu nie ma q/2 bo wyzej (nad ***
		# jest spierwiastkowane RMS od razu.
            qRMS_dict_dla_skali = dict(zip(qorder, qRMS_dla_skali))

            Fq_dla_skali = []
            for q in qorder:
	        if -0.001 < q < 0.001:
                    Fq_dla_skali.append(numpy.exp(0.5*numpy.mean(numpy.log(qRMS_dict_dla_skali[q]))))
                else:
                    Fq_dla_skali.append(numpy.mean(qRMS_dict_dla_skali[q])**(1.0/float(q)))

            Fq_dict_dla_skali = dict(zip(qorder,Fq_dla_skali))

            RMS_scale.append(RMS_dla_skali)
            qRMS_scale.append(qRMS_dict_dla_skali)
            Fq_scale.append(Fq_dict_dla_skali)

        RMS_dict_scale = dict(zip(scales,RMS_scale))
        qRMS_dict_scale = dict(zip(scales,qRMS_scale))
        Fq_dict_scale = dict(zip(scales,Fq_scale))

        Fq_m.append(Fq_dict_scale)
        RMS_m.append(RMS_dict_scale)
        qRMS_m.append(qRMS_dict_scale)

    Fq = dict(zip([emd],Fq_m))
    qRMS = dict(zip([emd],qRMS_m))
    RMS = dict(zip([emd],RMS_m))

    Hq_m = []
    tq_m = []
    hq_m = []
    Dq_m = []
    for m in [emd]:
        Fq_qorder = dict(zip(qorder,[sorted([Fq[m][i][j] for i in scales]) for j in qorder]))
        Hq_qorder = []
        tq_qorder = []
        for q in qorder:
            C = numpy.polyfit(numpy.log2(numpy.array(scales)), numpy.log2(numpy.array(Fq_qorder[q])),1)
            Hq_qorder.append(C)
            tq_qorder.append(C.tolist()[0] * q - 1.0)
        Hq_m.append(dict(zip(qorder,Hq_qorder)))
        tq_m.append(dict(zip(qorder,tq_qorder)))

        hq_qorder = numpy.diff(tq_qorder)/numpy.diff(qorder)
        hq_m.append(dict(zip(qorder[:-1],hq_qorder.tolist())))
        Dq_m.append(dict(zip(qorder[:-1],(numpy.array(qorder[:-1])*hq_qorder) - numpy.array(tq_qorder[:-1]))))

    Hq = dict(zip([emd], Hq_m)) # q-order Hurst exponent
    tq = dict(zip([emd], tq_m)) # q-order mass exponent
    hq = dict(zip([emd], hq_m)) # q-order singularity exponent (Hoelder exponent)
    Dq = dict(zip([emd], Dq_m)) # q-order singularity dimension

    return {
	'data':data.tolist(),
	'BW':X.tolist(),
	'emd':emd,
	'scales':scales,
	'q':qorder,
	'Fq':Fq,
	'RMS':RMS,
	'qRMS':qRMS,
	'Hq':Hq,
	'tq':tq,
	'hq':hq,
	'Dq':Dq,
	}


################
### The below located function is here for some unknown historical reason
### anything it caclulates is redundant and can be found in the fmfssc
### function above
################
def mfdfa(data, polyorder=[1], scales=[], qorder=list(range(-15,16)),
           with_zero_q=True, shuffle=False, raw_data=False,
           ndiff=None, walking_window=None, quiet=True):
    """
    ***
    * based on
    * 1. "Introduction to multifractal detrended fluctuation analysis in Matlab",
    *  	E. A. F. Ihlen, Front. Physio. 3, 141 (2012)
    * 2. "Multifractal detrended fluctuation analysis of nonstationary time series",
    *	J. W. Kantelhardt, S. A. Zschiegner, E. Koscielny-Bundec, S. Havlin,
    *	A. Bunde and H. E. Stanley, Phys. A 316, 87 (2002)
    ***

    Funkcja obliczajaca szereg parametrow charakteryzujacych szereg czasowy.
    Analiza oparta jest na zalozeniu multiraktalnych wlasnosci szeregu. W wyniku
    obliczany jest wykladnik Hursta rzedu 'q' (q-order Hurst exponent).

    INPUT:
      data    - dane w postaci listy lub numpy.array zawierajace badany szereg czasowy
      	        (1D, zakladamy uporzadkowanie w czasie);
      polyorder - lista zawierajaca rzad (rzedy) wielomianow dopasowywanych do danych
      		w danym oknie; domyslnie dopasowanie jest liniowe;
      scales  - lista zawierajaca wielkosci okien branych w analizie do obliczania
      		wykladnika Hursta. Powinny zawierac potegi liczby 2;
		jezeli lista jest pusta (domyslnie) skale obliczane sa automatycznie;
      qorder  - lista wartosci dla ktorych obliczany jest RMS rzedu q oraz
      		odpowiadajacy mu wykladnik Hursta rzedu q;
		domyslnie qorder=range(-5,6,2);
      with_zero_q - wartosc decydujaca czy obliczc wykladnik Hursta (i cala reszte)
                    dla q = 0;
		    True - (domyslnie) oblicza F(q=0) jezeli takowa zawarta jest w liscie qorder,
		    False - pomija (lub usuwa) z listy qorder wartosc q=0;
      shuffle - bazujemy na danych oryginalnych, czy przetasowanych losowo;
      		ma to na celu sprawdzenie, czy multifraktalne charakterystyki
		pochodza z korelacji dalekozasiegowych, czy z szerokiego
		rozkladu gestosci prawdopodobienstwa (jezeli multifraktalnosc
		bierze sie z korelacji, to zniknie):
		dla pomieszanych danych;
      		True - miesza losowo dane przed obliczaniem charakterystyk,
		False - (domyslnie) pracuje na oryginalnych danych;
      ndiff - roznicowanie danych, obliczanie przyrostow oryginalnych danych,
	      typowa technika osiagania stacjonarnosci z danych niestacjonarnych:
	      None - (domyslnie) pracuje na oryginalnych danych
	      1,2,3,... - stopien roznicowania
      raw_data - definiuje dane ktore poddajemy analizie
                 True - oblicza charakterystyki dla danych oryginalnych (po odjeciu sredniej)
                 False - (domyslnie) oblicza charakterystyki dla danych scalkowanych (cumsum)
      walking_window - definiuje nalozenie sie segmentow podczas obliczania Fq na siebie
                       None - (domyslnie) segmenty rozdzielne
		       1,2,3,... - nakladajace sie segmenty, jest to krok z jakim segment ma
		       podrozowac po danych

    OUTPUT:

      Slownik
      {'Fq':Fq, 'RMS':RMS, 'qRMS':qRMS, 'm':polyorder,
      	'scales':scales, 'q':qorder, 'data':data, 'Hq':Hq,
	'tq':tq, 'hq':hq, 'Dq':Dq}

      dane wejsciowe:
      m      - l

      ista rzedow wielomianow uzytych do dopasowania polyfit-em: dict['m']
      scales - lista wielkosci okien (skal) uzytych w analizie, skala
      	       16 <= s <= max(len(data)), zawsze potega liczby 2: dict['scales']
      q      - lista wykladnikow uzytych do okreslenia rzedu qRMS, Fq i Hq; nie
      	       zawiera liczby 0: dict['q']
      data   - lista trzymajaca dane: dict['data']

      dane z analizy:
      RMS    - slownik; trzyma lokalny trend dopasowany wielomianem 'm'-tego rzedu dla
      	       danej skali 's' dla calosci danych (obcietych do wielokrotnosci dlugosci
	       okna):  dict['RMS'][m][s]
      qRMS   - slownik; zwraca lokalny trend rzedu 'q' - funkcje probkojaca gdzie w
               szeregu wystepuja duze (duze dodatnie q) i male (ujemne q) fluktuacje
	       dla dopasowania wielomianem rzedu 'm', okien o skali (wielkosci) 's':
	       dict['qRMS'][m][s][q]
      Fq     - slownik; funkcja fluktuacji rzedu 'q' dla danego rzedu dopasowania
	       wielomianu 'm' i skali 's' (wielkosci okien): dict['Fq'][m][s][q]
      Hq     - slownik; zwraca liniowe (y=ax+b) dopasowanie do log(Fq) vs log(scales)
      	       dla rzedow q i m (jak wyzej): wykladnik Hursta rzedu q jest pierwszym
	       elementem dopasowania: Hq = a = dict['Hq'][m][q][0], b = dict['Hq'][m][q][1]
      tq     - slownik; wykladnik masowy rzedu q (q-order mass exponent): dict['tq'][m][q]
      hq     - slownik; natezenie osobliwosci rzedu q lub wykladnik Hoersta (q-order
      	       singularity strenght): dict['hq'][m][q]
      Dq     - slownik; wymiar osobliwosci rzedu q (q-order singularity dimension):
               dict['Dq'][m][q]

    EXAMPLES:
      zakladamy, ze szereg mamy w liscie 'dane' (moze ty byc lista pythonowa lub
      ndarray numpy)

      sage: dict_mfdfa1 = mfdfa1(dane)
        zwroci nam slownik wielkosci dla m = 1, wielkosci okien scales=[16,32,64,128] oraz
        wykladnikow dopasowania RMS rzedu q=[-3,-2,-1,1,2,3]

      sage: max_range = numpy.floor(numpy.log2(dane/10)) + 2
      sage: _fit = [1,2,3]
      sage: _sca = [2**i for i in range(4,max_range)]
      sage: _q = range(-9,0)+range(1,10)
      sage: dict_mfdfa1 = mfdfa1(dane, polyorder=_fit, scales=_sca, qorder=_q)
        zwroci nam slownik wielkosci dla m = [1,2,3] (liniowe, kwadratowe i szescianowe
	dopasowanie), wielkosci okien scales=[16,32,64,128,...,MAX2] (od 16 az do okna
	o szerokosci mieszczacej sie tylko raz w obrebie danych), oraz wykladnikow
	dopasowania RMS rzedu q=[-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9]

      sage: dict_mfdfa1 = mfdfa1(dane, polyorder=[2], scales=[32,64,128], qorder=range(-5,5), with_zero_q=True)
        zwroci slownik dla wszystkich podanych wielkosci wraz z parametrami oblicznonymi
	dla q=0
	"""

    import numpy, warnings

    # numpy.array
    X = numpy.array(data)

    # roznicowanie danych (do stacjonarnosci)
    if ndiff != None:
      X = numpy.diff(X, n=ndiff)

    # charakterystyki dla pomieszanych danych
    if shuffle:
      numpy.random.shuffle(X)

    # raw or integrated data
    if raw_data:
      X = X - numpy.mean(X)
    else:
      X = numpy.cumsum(X - numpy.mean(X))

    # spr czy jest 0 w potegach i usunac na zyczenie...
    if not with_zero_q and 0 in qorder: qorder.remove(0)

    #automatic scales
    if scales == []:
        scales = [2**i for i in range(4,int(numpy.floor(numpy.log2(len(data)/10))) + 2)]

    RMS_m = []
    qRMS_m = []
    Fq_m = []
    for m in polyorder:
        """rzad wielomianu do dopasowania"""

        RMS_scale = []
        qRMS_scale = []
        Fq_scale = []
        for ns in scales:
            if walking_window == None:
                step = ns
                segments = int(numpy.floor(len(X)/ns))
            else:
                assert(0 < walking_window < ns)
                step = walking_window
                segments = int(numpy.floor(len(X)/step)) - ns - 1

            if not quiet:
                print("len(X) %d, step %d, segments %d, step*segments %d"%(len(X), step, segments, step*segments))

            RMS_dla_skali = []
            for v in range(segments):
                idx_start = v * step
                idx_stop  = idx_start + ns

                C = numpy.polyfit(range(idx_start,idx_stop), X[idx_start:idx_stop],m)
                fit = numpy.poly1d(C)
                _b = numpy.sqrt(numpy.mean((X[idx_start:idx_stop] - fit(range(idx_start,idx_stop)))**2))
                # ***
                RMS_dla_skali.append(_b)

            qRMS_dla_skali = []
            for q in qorder:
                if -0.001 < q < 0.001:
                    qRMS_dla_skali.append(numpy.array(RMS_dla_skali)**2)
                else:
                    qRMS_dla_skali.append(numpy.array(RMS_dla_skali)**(float(q)))
                    # Tu lekkie wyjasnienie na przyszlosc: tu nie ma q/2 bo wyzej (nad ***
                    # jest spierwiastkowane RMS od razu.
            qRMS_dict_dla_skali = dict(zip(qorder, qRMS_dla_skali))

            Fq_dla_skali = []
            for q in qorder:
                if -0.001 < q < 0.001:
                    Fq_dla_skali.append(numpy.exp(0.5*numpy.mean(numpy.log(qRMS_dict_dla_skali[q]))))
                else:
                    Fq_dla_skali.append(numpy.mean(qRMS_dict_dla_skali[q])**(1.0/float(q)))
            Fq_dict_dla_skali = dict(zip(qorder,Fq_dla_skali))

            RMS_scale.append(RMS_dla_skali)
            qRMS_scale.append(qRMS_dict_dla_skali)
            Fq_scale.append(Fq_dict_dla_skali)

        RMS_dict_scale = dict(zip(scales,RMS_scale))
        qRMS_dict_scale = dict(zip(scales,qRMS_scale))
        Fq_dict_scale = dict(zip(scales,Fq_scale))

        Fq_m.append(Fq_dict_scale)
        RMS_m.append(RMS_dict_scale)
        qRMS_m.append(qRMS_dict_scale)

    Fq = dict(zip(polyorder, Fq_m))
    qRMS = dict(zip(polyorder, qRMS_m))
    RMS = dict(zip(polyorder, RMS_m))

    Hq_m = []
    tq_m = []
    hq_m = []
    Dq_m = []
    for m in polyorder:
        Fq_qorder = dict(zip(qorder, [sorted([Fq[m][i][j] for i in scales]) for j in qorder]))
        Hq_qorder = []
        tq_qorder = []
        for q in qorder:
            C = numpy.polyfit(numpy.log2(numpy.array(scales)),numpy.log2(numpy.array(Fq_qorder[q])),1)
            Hq_qorder.append(C)
            tq_qorder.append(C.tolist()[0] * q - 1.0)
        Hq_m.append(dict(zip(qorder,Hq_qorder)))
        tq_m.append(dict(zip(qorder,tq_qorder)))

        hq_qorder = numpy.diff(tq_qorder)/numpy.diff(qorder)
        hq_m.append(dict(zip(qorder[:-1],hq_qorder.tolist())))
        Dq_m.append(dict(zip(qorder[:-1],(numpy.array(qorder[:-1])*hq_qorder) - numpy.array(tq_qorder[:-1]))))

    Hq = dict(zip(polyorder,Hq_m)) # q-order Hurst exponent
    tq = dict(zip(polyorder,tq_m)) # q-order mass exponent
    hq = dict(zip(polyorder,hq_m)) # q-order singularity exponent (Hoelder exponent)
    Dq = dict(zip(polyorder,Dq_m)) # q-order singularity dimension

    return {
        'data':data.tolist(),
        'BW':X.tolist(),
        'm':polyorder,
        'scales':scales,
        'q':list(qorder),
        'Fq':Fq,
        'RMS':RMS,
        'qRMS':qRMS,
        'Hq':Hq,
        'tq':tq,
        'hq':hq,
        'Dq':Dq,
        }


if __name__ == "__main__":
    print("please import\nimport mfa")
