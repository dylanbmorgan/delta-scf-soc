#!/usr/bin/env python3

h_bar = 1

azimuthal = [1]
ang_mom = [-1, 0, 1]
spin = [-1 / 2, 1 / 2]
s = 0.5

lz_szs = []
l_up_s_downs = []
l_down_s_ups = []
totals = []


for l in azimuthal:
    for m_l in ang_mom:
        for m_s in spin:
            lz_sz = h_bar**2 * m_l * m_s
            lz_szs.append(lz_sz)

            l_up_s_down = (
                0.5
                * h_bar**2
                * (l**2 + l - m_l**2 + m_l) ** 0.5
                * (s**2 + s - m_s**2 - m_s) ** 0.5
            )
            l_up_s_downs.append(l_up_s_down)

            l_down_s_up = (
                0.5
                * h_bar**2
                * (l**2 + l - m_l**2 - m_l) ** 0.5
                * (s**2 + s - m_s**2 + m_s) ** 0.5
            )
            l_down_s_ups.append(l_down_s_up)

            total = lz_sz + l_up_s_down + l_down_s_up
            totals.append(total)

print(lz_szs)
print(l_up_s_downs)
print(l_down_s_ups)
print(totals)
