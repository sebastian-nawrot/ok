from datetime import timedelta, datetime
from pprint import pprint

import math
import numpy as np
import sys
import random


# PARAMETRY
ilosc_wierzcholkow = 20
minimum_krawedzi = 6
maksimum_krawedzi = 30
minimalna_waga = 1
maksymalna_waga = 100

czas_dzialania_algorytmu = 60 * 4
ilosc_mrowek = 100
ilosc_dopuszczalnych_rozwiazan = 1
stopien_parowania_feromonow = 0.1

szansa_uzycia_feromonow = 0.01
wzrost_szansy_uzycia_feromonow = 0.001

granica_wygladzania = 40
wartosc_wygladzania = 20

wartosc_alfa = 1
wartosc_beta = 0



def generuj_losowy_graf():
  graf = np.zeros((ilosc_wierzcholkow, ilosc_wierzcholkow), dtype=int)
  for i in range(1, ilosc_wierzcholkow):
    graf[i-1][i] = graf[i][i-1] = random.randint(minimalna_waga, maksymalna_waga)

  for i in range(ilosc_wierzcholkow):
    do_uzupelnienia = minimum_krawedzi - len(graf[i].nonzero()[0])
    if do_uzupelnienia > 0:
      for _ in range(do_uzupelnienia):
        pozostale = set(range(ilosc_wierzcholkow)) - set(graf[i].nonzero()[0]) - {i}
        while pozostale:
          j = random.choice(tuple(pozostale))
          if len(graf[:,j].nonzero()[0]) < maksimum_krawedzi:
            graf[i][j] = graf[j][i] = random.randint(minimalna_waga, maksymalna_waga)
            break
          pozostale.remove(j)
        else:
          raise Exception('NIE MOŻNA UTWORZYĆ GRAFU')

  assert all(minimum_krawedzi <= len(x.nonzero()[0]) <= maksimum_krawedzi for x in graf)
  return graf


def wylicz_koszt(graf, sciezka):
  wagi = np.empty(len(sciezka) - 1, dtype=int)
  for i in range(0, len(sciezka) - 1):
    j, k = sciezka[i:i+2]
    wagi[i] = graf[j, k]
    if (i + 1) % 5 == 0:
      wagi[i] = wagi[i] + sum(wagi[i-2:i+1]) * 2 * len(graf[k].nonzero()[0])
  return sum(wagi)


def start_mrowki(graf, feromony, prawdopodobienstwa):
  sciezka = np.empty(ilosc_wierzcholkow, dtype=int)
  indeks, maksymalny_indeks = 0, ilosc_wierzcholkow - 1
  sciezka[0] = random.randrange(0, ilosc_wierzcholkow)
  uzycie_fermonow = random.random() < szansa_uzycia_feromonow
  niedowiedzone = set(range(ilosc_wierzcholkow)) - {sciezka[0]}

  while niedowiedzone:
    sasiedzi = graf[sciezka[indeks]].nonzero()[0]

    niedowiedzeni_sasiedzi = tuple(niedowiedzone & set(sasiedzi))
    if niedowiedzeni_sasiedzi:
      if uzycie_fermonow:
        prawdopodobienstwa_niedowiedzonych = np.zeros(ilosc_wierzcholkow)
        prawdopodobienstwa_niedowiedzonych[niedowiedzeni_sasiedzi, ] = \
          prawdopodobienstwa[sciezka[indeks], niedowiedzeni_sasiedzi]

        nastepny = random.choices(list(range(0, ilosc_wierzcholkow)),
                                  weights=prawdopodobienstwa_niedowiedzonych)[0]
      else:
        nastepny = random.choice(niedowiedzeni_sasiedzi)
    else:
      if uzycie_fermonow:
        nastepny = random.choices(list(range(0, ilosc_wierzcholkow)),
                          weights=prawdopodobienstwa[sciezka[indeks]])[0]
      else:
        nastepny = random.choice(sasiedzi)

    if nastepny in niedowiedzone:
      niedowiedzone.remove(nastepny)
    indeks += 1
    if indeks > maksymalny_indeks:
      sciezka.resize(maksymalny_indeks + 11)
      maksymalny_indeks += 10
    sciezka[indeks] = nastepny


  sciezka.resize(indeks + 1)
  return wylicz_koszt(graf, sciezka), sciezka



current_values = []
best_values = []

if __name__ == "__main__":
  macierz_grafu = generuj_losowy_graf()

  macierz_feromonow = np.zeros((ilosc_wierzcholkow, ilosc_wierzcholkow), dtype=float)
  macierz_feromonow[macierz_grafu.nonzero()] = 1

  macierz_prawdopodobienstwa = np.zeros((ilosc_wierzcholkow, ilosc_wierzcholkow), dtype=float)
  macierz_prawdopodobienstwa = np.copy(macierz_feromonow)
  

  najlepsze_rozwiazanie = None
  pierwsza_iteracja = True
  stop = datetime.now() + timedelta(seconds=czas_dzialania_algorytmu)
  while datetime.now() < stop:
    sciezki = []
    for _ in range(ilosc_mrowek):
      sciezki.append(start_mrowki(macierz_grafu,
                                  macierz_feromonow,
                                  macierz_prawdopodobienstwa))

    sciezki = sorted(sciezki, key=lambda x: x[0])
    najlepsze_sciezki = sciezki[:ilosc_dopuszczalnych_rozwiazan]

    roznica = najlepsze_sciezki[-1][0] - najlepsze_sciezki[0][0]
    for koszt, sciezka in najlepsze_sciezki:
      if roznica:
        moc_feromonu = (koszt - najlepsze_sciezki[0][0]) / roznica * 0.9
      else:
        moc_feromonu = 1

      for j in range(0, len(sciezka) - 1):
        k, l = sciezka[j:j+2]
        macierz_feromonow[k, l] += 1 - moc_feromonu

    macierz_feromonow *= 1 - stopien_parowania_feromonow

    macierz_prawdopodobienstwa = (macierz_feromonow ** wartosc_alfa)
    macierz_prawdopodobienstwa *= (macierz_grafu / 1) ** wartosc_beta

    for wiersz in macierz_feromonow:
      if np.where(wiersz > granica_wygladzania)[0].size:
        minimum = wiersz[wiersz.nonzero()].min()
        for i, x in enumerate(wiersz):
          if x > 0:
            wiersz[i] = minimum * (1 + math.log(x / minimum, wartosc_wygladzania))


    szansa_uzycia_feromonow += wzrost_szansy_uzycia_feromonow

    current_values.append(sciezki[0][0])

    if not najlepsze_rozwiazanie or sciezki[0][0] < najlepsze_rozwiazanie:
      najlepsze_rozwiazanie = sciezki[0][0]
      best_values.append(najlepsze_rozwiazanie)
      print(najlepsze_rozwiazanie)
    else:
      best_values.append(best_values[-1])
