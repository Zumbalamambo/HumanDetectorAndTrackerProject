def overlaps(trcks,coordinates):
    percentages=[]
    if len(trcks)>len(coordinates):
        print "a person leaves"
    if len(trcks)<len(coordinates):
        print "new person enterd"

    for k in trcks:
        for l in coordinates:

            area1 = abs(k[0] - k[2]) * abs(k[1] - k[3])

            cod1 = abs(l[0] - l[2]) * abs(l[1] - l[3])


            is1 = abs(max(k[0], l[0]) - min(k[2], l[2])) * abs(
            max(k[1], l[1]) - min(k[3], l[3]))
            perc1 = (float(is1) / (area1 + cod1 - is1)) * 100
            if perc1>75:
                print "success",k,l
            percentages.append(round(perc1,2))


    print percentages

overlaps([[231, 133, 368, 436], [459, 136, 578, 424]],[[233, 131, 372, 438], [456, 137, 574, 421]])
