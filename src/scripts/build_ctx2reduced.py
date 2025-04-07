# Existing mappings (example excerpt; in practice these would be loaded from your JSON)
fs2reduced = {
    "4": 1, "5": 2, "8": 3, "10": 4, "11": 5, "12": 6, "13": 7, "14": 8,
    "15": 9, "16": 10, "17": 11, "18": 12, "24": 13, "26": 14, "28": 15, "30": 16,
    "31": 17, "43": 18, "44": 19, "47": 20, "49": 21, "50": 22, "51": 23, "52": 24,
    "53": 25, "54": 26, "58": 27, "60": 28, "62": 29, "63": 30, "80": 31, "85": 32,
    "1001": 33, "1002": 34, "1003": 35, "1005": 36, "1006": 37, "1007": 38, "1008": 39,
    "1009": 40, "1010": 41, "1011": 42, "1012": 43, "1013": 44, "1014": 45, "1015": 46,
    "1016": 47, "1017": 48, "1018": 49, "1019": 50, "1020": 51, "1021": 52, "1022": 53,
    "1023": 54, "1024": 55, "1025": 56, "1026": 57, "1027": 58, "1028": 59, "1029": 60,
    "1030": 61, "1031": 62, "1032": 63, "1033": 64, "1034": 65, "1035": 66,
    "2001": 67, "2002": 68, "2003": 69, "2005": 70, "2006": 71, "2007": 72, "2008": 73,
    "2009": 74, "2010": 75, "2011": 76, "2012": 77, "2013": 78, "2014": 79, "2015": 80,
    "2016": 81, "2017": 82, "2018": 83, "2019": 84, "2020": 85, "2021": 86, "2022": 87,
    "2023": 88, "2024": 89, "2025": 90, "2026": 91, "2027": 92, "2028": 93, "2029": 94,
    "2030": 95, "2031": 96, "2032": 97, "2033": 98, "2034": 99, "2035": 100
}

ctx_names2fs = {
    "ctx-lh-unknown": 1000,
    "ctx-lh-bankssts": 1001,
    "ctx-lh-caudalanteriorcingulate": 1002,
    "ctx-lh-caudalmiddlefrontal": 1003,
    "ctx-lh-corpuscallosum": 1004,
    "ctx-lh-cuneus": 1005,
    "ctx-lh-entorhinal": 1006,
    "ctx-lh-fusiform": 1007,
    "ctx-lh-inferiorparietal": 1008,
    "ctx-lh-inferiortemporal": 1009,
    "ctx-lh-isthmuscingulate": 1010,
    "ctx-lh-lateraloccipital": 1011,
    "ctx-lh-lateralorbitofrontal": 1012,
    "ctx-lh-lingual": 1013,
    "ctx-lh-medialorbitofrontal": 1014,
    "ctx-lh-middletemporal": 1015,
    "ctx-lh-parahippocampal": 1016,
    "ctx-lh-paracentral": 1017,
    "ctx-lh-parsopercularis": 1018,
    "ctx-lh-parsorbitalis": 1019,
    "ctx-lh-parstriangularis": 1020,
    "ctx-lh-pericalcarine": 1021,
    "ctx-lh-postcentral": 1022,
    "ctx-lh-posteriorcingulate": 1023,
    "ctx-lh-precentral": 1024,
    "ctx-lh-precuneus": 1025,
    "ctx-lh-rostralanteriorcingulate": 1026,
    "ctx-lh-rostralmiddlefrontal": 1027,
    "ctx-lh-superiorfrontal": 1028,
    "ctx-lh-superiorparietal": 1029,
    "ctx-lh-superiortemporal": 1030,
    "ctx-lh-supramarginal": 1031,
    "ctx-lh-frontalpole": 1032,
    "ctx-lh-temporalpole": 1033,
    "ctx-lh-transversetemporal": 1034,
    "ctx-lh-insula": 1035,
    "ctx-rh-unknown": 2000,
    "ctx-rh-bankssts": 2001,
    "ctx-rh-caudalanteriorcingulate": 2002,
    "ctx-rh-caudalmiddlefrontal": 2003,
    "ctx-rh-corpuscallosum": 2004,
    "ctx-rh-cuneus": 2005,
    "ctx-rh-entorhinal": 2006,
    "ctx-rh-fusiform": 2007,
    "ctx-rh-inferiorparietal": 2008,
    "ctx-rh-inferiortemporal": 2009,
    "ctx-rh-isthmuscingulate": 2010,
    "ctx-rh-lateraloccipital": 2011,
    "ctx-rh-lateralorbitofrontal": 2012,
    "ctx-rh-lingual": 2013,
    "ctx-rh-medialorbitofrontal": 2014,
    "ctx-rh-middletemporal": 2015,
    "ctx-rh-parahippocampal": 2016,
    "ctx-rh-paracentral": 2017,
    "ctx-rh-parsopercularis": 2018,
    "ctx-rh-parsorbitalis": 2019,
    "ctx-rh-parstriangularis": 2020,
    "ctx-rh-pericalcarine": 2021,
    "ctx-rh-postcentral": 2022,
    "ctx-rh-posteriorcingulate": 2023,
    "ctx-rh-precentral": 2024,
    "ctx-rh-precuneus": 2025,
    "ctx-rh-rostralanteriorcingulate": 2026,
    "ctx-rh-rostralmiddlefrontal": 2027,
    "ctx-rh-superiorfrontal": 2028,
    "ctx-rh-superiorparietal": 2029,
    "ctx-rh-superiortemporal": 2030,
    "ctx-rh-supramarginal": 2031,
    "ctx-rh-frontalpole": 2032,
    "ctx-rh-temporalpole": 2033,
    "ctx-rh-transversetemporal": 2034,
    "ctx-rh-insula": 2035
}

# Initialize the new mappings for cortex names to reduced indices.
ctx_names2reduced = {}
# We'll build the inverse mapping as well.
reduced2ctx_names = {}

# Iterate over each cortical region in ctx_names2fs.
for ctx_name, fs_id in ctx_names2fs.items():
    # Convert fs_id to string to check in fs2reduced
    fs_id_str = str(fs_id)
    if fs_id_str in fs2reduced:
        # If fs_id is present in fs2reduced, use that mapping.
        reduced_index = fs2reduced[fs_id_str]
    else:
        continue  # Skip if fs_id is not in fs2reduced

    # Update the new mappings.
    ctx_names2reduced[ctx_name] = reduced_index
    reduced2ctx_names[str(reduced_index)] = ctx_name

# Print the mappings for verification.
print(ctx_names2reduced)
print(reduced2ctx_names)