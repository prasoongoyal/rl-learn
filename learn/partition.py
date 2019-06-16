class Partition:
    def __init__(self):
        self.side_labels = {}

        self.side_labels['249'] = {}
        self.side_labels['249']['C'] =    list(range(0,1151)) + list(range(2697,2868)) + list(range(6285,6321)) + list(range(8267,8800))
        self.side_labels['249']['L'] =    list(range(1152,2696))
        self.side_labels['249']['R'] =    list(range(2869,6284)) + list(range(6322,8112))

        self.side_labels['275'] = {}
        self.side_labels['275']['C'] =    list(range(0,878)) + list(range(5825,7414)) + list(range(9722,9773))
        self.side_labels['275']['L'] =    list(range(7415,9721)) + list(range(9774,11025))
        self.side_labels['275']['R'] =    list(range(879,5824))

        self.side_labels['281'] = {}
        self.side_labels['281']['C'] =    list(range(0,1752)) + list(range(6473,9180)) + list(range(9580,10338))
        self.side_labels['281']['L'] =    list(range(10339,10755))
        self.side_labels['281']['R'] =    list(range(1754,6472)) + list(range(9199,9579))

        self.side_labels['283'] = {}
        self.side_labels['283']['C'] =    list(range(0,952)) + list(range(5701,8616)) + list(range(9794,9968))
        self.side_labels['283']['L'] =    list(range(9969,12207)) + list(range(12208,12557))
        self.side_labels['283']['R'] =    list(range(953,5700)) + list(range(8617,9793))

        self.side_labels['300'] = {}
        self.side_labels['300']['C'] =    list(range(0,847)) + list(range(6722,8454)) + list(range(8654,8805))
        self.side_labels['300']['L'] =    list(range(8455,8652))
        self.side_labels['300']['R'] =    list(range(848,6721)) + list(range(8806,9596))

        self.side_labels['324'] = {}
        self.side_labels['324']['C'] =    list(range(0,704)) + list(range(2062,3121)) + list(range(6454,6834))
        self.side_labels['324']['L'] =    list(range(706,2061))
        self.side_labels['324']['R'] =    list(range(3122,6453))

        self.side_labels['546'] = {}
        self.side_labels['546']['C'] =    list(range(0,1091)) + list(range(7621,7904)) + list(range(8433,9636))
        self.side_labels['546']['L'] =    list(range(1092,7620)) + list(range(7905,8432))
        self.side_labels['546']['R'] =    []

        self.side_labels['559'] = {}
        self.side_labels['559']['C'] =    list(range(0,1025)) + list(range(7292,7780))
        self.side_labels['559']['L'] =    list(range(1026,7291))
        self.side_labels['559']['R'] =    []

        self.side_labels['563'] = {}
        self.side_labels['563']['C'] =    list(range(0,921)) + list(range(5033,6791)) + list(range(7643,8169))
        self.side_labels['563']['L'] =    list(range(922,5031))
        self.side_labels['563']['R'] =    list(range(6793,7641)) + list(range(8171,9294))

        self.side_labels['658'] = {}
        self.side_labels['658']['C'] =    list(range(0,1423)) + list(range(5962,7709))
        self.side_labels['658']['L'] =    []
        self.side_labels['658']['R'] =    list(range(1425,5961))

        self.side_labels['958'] = {}
        self.side_labels['958']['C'] =    list(range(0,1356)) + list(range(7289,8871))
        self.side_labels['958']['L'] =    []
        self.side_labels['958']['R'] =    list(range(1358,7288))

        self.side_labels['1199'] = {}
        self.side_labels['1199']['C'] =    list(range(0,893)) + list(range(4718,6679)) + list(range(7375,7695))
        self.side_labels['1199']['L'] =    []
        self.side_labels['1199']['R'] =    list(range(894,4716)) + list(range(6680,7374))

        self.side_labels['1256'] = {}
        self.side_labels['1256']['C'] =    list(range(0,2512))
        self.side_labels['1256']['L'] =    []
        self.side_labels['1256']['R'] =    list(range(2513,9743))

        self.side_labels['1289'] = {}
        self.side_labels['1289']['C'] =    list(range(0,924)) + list(range(5466,7791))
        self.side_labels['1289']['L'] =    list(range(925,5464))
        self.side_labels['1289']['R'] =    []

        self.side_labels['1340'] = {}
        self.side_labels['1340']['C'] =    list(range(0,1686))
        self.side_labels['1340']['L'] =    []
        self.side_labels['1340']['R'] =    list(range(1687,7304))

        self.side_labels['1381'] = {}
        self.side_labels['1381']['C'] =    list(range(0,852)) + list(range(4763,8719))
        self.side_labels['1381']['L'] =    list(range(853,4761))
        self.side_labels['1381']['R'] =    list(range(8720,10766))

        self.side_labels['1383'] = {}
        self.side_labels['1383']['C'] =    list(range(0,776)) + list(range(4664,6489))
        self.side_labels['1383']['L'] =    list(range(777,4663)) + list(range(6490,7360))
        self.side_labels['1383']['R'] =    []

        self.side_labels['1431'] = {}
        self.side_labels['1431']['C'] =    list(range(0,757)) + list(range(4023,5179)) + list(range(11689,12145))
        self.side_labels['1431']['L'] =    list(range(758,4022))
        self.side_labels['1431']['R'] =    list(range(5180,11687))

        self.side_labels['1458'] = {}
        self.side_labels['1458']['C'] =    list(range(0,1925)) + list(range(6517,7906))
        self.side_labels['1458']['L'] =    list(range(1926,6516))
        self.side_labels['1458']['R'] =    []

        self.side_labels['1473'] = {}
        self.side_labels['1473']['C'] =    list(range(0,1639))
        self.side_labels['1473']['L'] =    list(range(1640,8959))
        self.side_labels['1473']['R'] =    []

    def is_sublist_of(self, a, b):
        return all(elem in b for elem in a)

    def clip_id_to_side(self, clip_id):
        vidid = clip_id.split('/')[0]
        start = int(eval(clip_id.split('_')[-1][:-4]) * 50)
        end = start + 150
        curr_indices = range(start,end)
        if self.is_sublist_of(curr_indices, self.side_labels[vidid]['L']):
            return 'L'
        if self.is_sublist_of(curr_indices, self.side_labels[vidid]['R']):
            return 'R'
        if self.is_sublist_of(curr_indices, self.side_labels[vidid]['C']):
            return 'C'
        return None


