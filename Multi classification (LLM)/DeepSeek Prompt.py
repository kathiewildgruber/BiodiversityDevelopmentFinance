#%% DeepSeek LLM Labeling Script with Original Prompt (19 Labels)

import pandas as pd
from openai import OpenAI
import time

# === Setup API Client ===
client = OpenAI(
    api_key="",  # DeepSeek API Key to be inserted
    base_url="https://api.deepseek.com"
)

# === Config ===
input_csv_path = "inputs/crs_all_translated_2000-2023_unique_binarylabeled_1only_vfinal.csv"
output_csv_path = "outputs/crs_all_translated_2000-2023_unique_binarylabeled_multilabeled.csv"

label_columns = [
    "No_Biodiv", "Act_Pollut", "Act_Invasiv", "Act_Agri", "Act_ForestMgmt", "Act_Fish",
    "Act_WaterMgmt", "Act_OthMgmt", "Act_Protect", "Act_Resto", "Act_Undef",
    "Pol_Regul", "Pol_Know", "Pol_Undef",
    "Eco_CropGrass", "Eco_Forest", "Eco_SeaWater", "Eco_UrbInd", "Eco_Undef"
]

# === Load CSV input ===
df = pd.read_csv(input_csv_path, sep=';')
df['unique_text_identifier'] = df['unique_text_identifier'].astype(int)
df['text'] = df['text'].fillna('')
#df = df.head(50)  # Only process the first 50 rows

# === Parse helper ===
def parse_llm_output(output, label_columns):
    output = output.strip().lstrip('{').rstrip('}')
    parts = output.split(';')
    if len(parts) != 1 + len(label_columns):
        raise ValueError(f"Unexpected format: {output}")
    parsed = {
        "unique_text_identifier": int(float(parts[0].strip()))
    }
    for i, col in enumerate(label_columns):
        val = parts[i+1].strip()
        parsed[col] = '1' if val == '1' else ''
    return parsed

# === Prompt (remains unchanged except your logic is now positional) ===
base_prompt = """Classify Official Development Assistance (ODA) projects based on their textual descriptions regarding their potential positive impact on biodiversity. Use a 4-dimensional multi-classification framework, assigning classes for:
Dimension 0: No Biodiversity Project (Class 1)
Dimension 1: Targeted Actions – What is being done (Classes 2–11)
Dimension 2: Policy Tools – How is it supported (Classes 12–14)
Dimension 3: Ecosystem Type – Where does it take place (Classes 15–19)
###Input: You receive a .csv file with a column ‘text’ which contains textual project descriptions, an ‘id’ and ‘unique_text_identifier’ column as well as 19 multi-class columns (defined below).
###Output Format: Create a .csv file with the following columns:
‘unique_text_identifier’: Original unique text identifier
‘text’: original textual project description
’No_Biodiv’, … , ‘Eco_Undefined’: 19 columns (one for each class) each marked with ‘1’ if applicable, otherwise, leave cell blank


###Rules:
Do not modify project descriptions or unique text identifiers
Use only the 19 classes described below for labelling
Use only binary labels (‘1’ or empty)
There might be false positives (= projects with no biodiversity or environmental enhancement) in the data set. If you identify such a project, assign class 1 (‘No_Biodiv’)  and leave cells in classes 2–19 completely blank
If the project is a true positive (i.e. not assigned to class 1 ‘No_Biodiv’) evaluate each of dimensions 1–3. Assign at least one class per dimension. 
If no specific class (2-10, 12-13, or 15-18) fits, assign the fallback for that dimension (11 = ‘Act_Undef’, 14 = ’Pol_Undef’, 19 = ’Eco_Undef’) and leave all other cells in that dimension blank
If for one project you would assign only fallback classes (i.e. 11 = ‘Act_Undef’, 14 = ’Pol_Undef’, 19 = ’Eco_Undef’), label the project as class 1 (‘No_Biodiv’). Explanation: In such a case the project is too general to be considered a biodiversity project
Multi-classification framework:
Dimension 0: False Positives (Class 1)
1. No Biodiversity Project (‘No_Biodiv’)
Only for projects that do not apply in any way to class any class 2-10 or 12,13.


Dimension 1: Targeted Action (Categories 2–11)
2. Pollution Control & Waste Management (‘Act_Pollut’): Applies to all projects that concern sanitation, sewerage systems, water quality enhancement or assessment, or waste management treatment. Also includes the prevention, reduction, or elimination of pollution in soil, water, air, and atmosphere, pollution mitigation strategies for chemicals and plastics (e.g., switching from conventional to organic agriculture, recycling of materials)
3. Invasive Species Management (‘Act_Invasiv’): Includes all projects regarding the prevention, control, or removal of invasive alien species.
4. Sustainable Agriculture (‘Act_Agri’): Applies to all projects that concern sustainable practices in agriculture and farming, sustainable crop and livestock management, agroforestry, agrobiodiversity, pollinator-friendly, organic, small-scale, or climate-resilient agriculture, or innovative farming practices. If this class applies, class 15 (‘Eco_CropGrass’) must be assigned as well.
5. Sustainable forestry (‘Act_ForestMgmt’): Applies to all projects that concern sustainable forest management, or agroforestry. If this class applies, class 16 (‘Eco_Forest’) must be assigned as well.
6. Sustainable fishery and aquaculture (‘Act_Fish’): Applies to all projects that concern the controlled or sustainable farming of aquatic species (fish, shellfish, aquatic plants) and aquaculture. If this class applies, class 17 (‘Eco_SeaWater’) must be assigned as well.
7. Sustainable and Integrated Water Management (‘Act_WaterMgmt’): Applies to all projects that concern sustainable or reduced water usage and integrated or sustainable water management
8. Other Sustainable Resource Management (‘Act_OthMgmt’): Applies to all projects that concern overall sustainable resource management, not specified by categories 4-7. This includes for example sustainable, controlled wildlife trade, or sustainable mining, or ecotourism, or urban biodiversity enhancement
9. Protection and Conservation (‘Act_Protect’): Applies to all projects that concern the protection or conservation of ecosystems or species or habitats, e.g. via the  establishment, expansion, or management of protected areas, introduction of wildlife corridors, nature conservation, environmental protection.
10. Restoration (‘Act_Resto’): Applies to all projects that concern the restoration of degraded ecosystems or habitats or nature to a more functional or natural state. Includes improving biodiversity in urban areas. Not in scope: restoration of buildings or historical sites.
11. Undefined Action (‘Act_Undef’): Use only if no other targeted action class (2–10) applies.
Dimension 2: Policy Tools (Categories 12–14)
12. Policy, Regulatory and Governance Support (‘Pol_Regul’): Development or implementation of biodiversity-related policies, laws, institutions, (inter-) institutional committees or governance structures benefiting the environment. Not in scope: Too broad or general sustainability policy initiatives.
13. Awareness & Knowledge Building (‘Pol_Know’): Education, training, advice, community engagement, network building, and participatory approaches to biodiversity issues or sustainable practices.
14. Undefined Policy Tool (‘Pol_Undef’): Use only if no other policy tools class (12 or 13) apply.
Dimension 3: Ecosystem Type (Categories 15–19)
15. Cropland, Rangeland, Grassland, Arid Land (‘Eco_CropGrass’): Agricultural or grazing lands, including arid or semi-arid regions. Includes agroforestry.
16. Forest (‘Eco_Forest’): Forested areas, also includes projects with agroforestry components.
17. Sea and Freshwater (‘Eco_SeaWater’): Marine or freshwater ecosystems, including rivers, lakes, wetlands, marshes, oceans, and coastal zones.
18. Urban and Industrial (‘Eco_UrbInd’): Projects in built land such as cities or urban spaces, rural development, or industrial areas.
19. Undefined Ecosystem Type (‘Eco_Undef’): Use only if no other ecosystem types category (15–18) applies.

###Guidelines for Output: Use consistent formatting for responses, as defined below. Only provide the classification. Do not repeat or summarize the input post or add explanations. Only use the explicit information provided to you.

###Input Format:
Input = 
{‘unique_text_identifier’: <unique_text_identifier>; ‘text’: <text>}

### Output Format:
Return only the classification in this exact flat format, with no line breaks or extra spaces:

{<unique_text_identifier>;
<No_Biodiv>;
<Act_Pollut>;
<Act_Invasiv>;
<Act_Agri>;
<Act_ForestMgmt>;
<Act_Fish>;
<Act_WaterMgmt>;
<Act_OthMgmt>;
<Act_Protect>;
<Act_Resto>;
<Act_Undef>;
<Pol_Regul>;
<Pol_Know>;
<Pol_Undef>;
<Eco_CropGrass>;
<Eco_Forest>;
<Eco_SeaWater>;
<Eco_UrbInd>;
<Eco_Undef>
}

Use only `1` to indicate that a class applies, or `0` if it does not apply. The order of binary values must strictly follow this sequence:

No_Biodiv, Act_Pollut, Act_Invasiv, Act_Agri, Act_ForestMgmt, Act_Fish, Act_WaterMgmt, Act_OthMgmt, Act_Protect, Act_Resto, Act_Undef, Pol_Regul, Pol_Know, Pol_Undef, Eco_CropGrass, Eco_Forest, Eco_SeaWater, Eco_UrbInd, Eco_Undef

Do not include class names, quotes, explanations, or formatting (e.g., no JSON, no line breaks, no comments). Return only a single semicolon-separated line inside curly braces.


Examples:
Input: {2905761;funding of the un secretary generals highlevel panel on global sustainability and its secretariat}
Output: {2905761;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0}
Input: {4378706;the project concerns the rehabilitation of irrigated perimeters small and medium hydraulic}
Output: {4378706;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0}
Input:{999;solar panel installation project to increase energy access and reduce carbon emissions in rural districts}
Output: {999;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0}
Input:{99;restoration of historic buildings and cultural monuments}
Output: {99;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0}
Input: {9999999;distribution of hygiene kits and community workshops to reduce cholera outbreaks}
Output: {9999999;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0}
Input: {99999;mangrove restoration to improve coastal protection and carbon sequestration while supporting fish breeding}
Output: {99999;0;0;0;0;0;0;0;0;1;1;0;0;0;1;0;0;1;0;0}
Input: {2853367;support indigenous peoples in selected communities in the peruvian amazon in their efforts to improve sustainable forest management practices dedicated grant mechanism in peru}
Output: {2853367;0;0;0;0;1;0;0;0;0;0;0;0;0;1;0;1;0;0;0}
Input: {1474546;develop new policies and laws that recognize and support the key contributions of rural people to the processes of sustainable genetic resources management and improvement dynamic biodiversity conservation access and benefit sharing of genetic resources national policy development}
Output: {1474546;0;0;0;0;0;0;0;1;1;0;0;1;0;0;0;0;0;0;1}
Input: {3567845;integrated rural development with emphasis on sustainable small scale agriculture}
Output: {3567845;0;0;0;1;0;0;0;0;0;0;0;0;0;1;1;0;0;0;0}
Input: {3939413;theme economic equity justice human rights movements institutions programme review the development infrastructure and investment projects in the energy transport and mining sectors of georgia in order to highlight risks of corruption and noncompliance with eu environmental standards and empower and strengthen the capacities of local communities and civil society groups to protect their social land and environmental rights association green}
Output: {3939413;0;0;0;0;0;0;0;1;0;0;0;1;0;0;0;0;0;1;0}
Input: {1456985;rehabilitation of the river watershed strengthening dialogue two components to the project technical assistance and capacity building for institutions to support rehabilitation of selected areas within the watershed halt the process of environmental degradation and reverse the degradation of the forest canopy improving the economic situation and respecting the environment}
Output: {1456985;0;0;0;0;0;0;1;0;1;1;0;0;1;0;0;1;1;0;0}
Input: {4069203;small grants project global environment facility support the achievement of global environmental benefits through community based solutions that work in harmony with actions at local national and global levels}
Output: {4069203;0;0;0;0;0;0;0;0;0;0;1;1;0;0;0;0;0;0;1}
Input: {111847;bombay sewage disposal project improving water quality strengthen the capacity of the water supply and sewerage department of the municipal corporation management of the provision of sewerage services including slum dwellers disposal}
Output: {111847;0;1;0;0;0;0;1;0;0;0;0;0;0;1;0;0;0;1;0}
Input: {9999;Invasive species management in columbian forest}
Output: {9999;0;0;1;0;0;0;0;0;0;0;0;0;0;1;0;1;0;0;0}
Input: {999999;city-based biodiversity awareness campaign and training workshops for municipal planners on green infrastructure}
Output: {999999;0;0;0;0;0;0;0;0;0;1;0;0;1;0;0;0;0;1;0}
Input: {999999999;development and training in the field of water supply and sanitation in calcutta reducing wastewater}
Output: {999999;0;1;0;0;0;0;1;0;0;1;0;0;1;0;0;0;0;1;0}

Now: perform the labeling task as described above for each project from the input csv. Do not display anything in the console but just save the relevant labels for each project in the output csv
"""

# === Run classification for each row ===
results = []

for idx, row in df.iterrows():
    text = row['text']
    full_prompt = base_prompt + f"\nInput: {{{row['unique_text_identifier']};{text}}}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for labeling ODA projects using a strict 19-class framework."},
                {"role": "user", "content": full_prompt}
            ],
            stream=False
        )
        output = response.choices[0].message.content.strip()
        print(f"[{idx+1}/{len(df)}] Output: {output}")

        parsed = parse_llm_output(output, label_columns)
        label_row = {col: parsed.get(col, '') for col in label_columns}
        df.at[idx, 'unique_text_identifier'] = parsed.get("unique_text_identifier", row['unique_text_identifier'])

    except Exception as e:
        print(f"Error at row {idx}: {e}")
        label_row = {col: "ERROR" for col in label_columns}

    results.append((idx, label_row))
    time.sleep(0.1)

    # === Autosave every 10 rows ===
    if (idx + 1) % 10 == 0:
        temp_label_df = pd.DataFrame([r for i, r in results])
        temp_output_df = pd.concat([df.iloc[:idx + 1].copy(), temp_label_df], axis=1)
        temp_output_df.to_csv(output_csv_path, index=False)
        print(f"Autosaved progress at row {idx + 1}")

# === Final save to CSV ===
# Rebuild label DataFrame with correct row indices
ordered_results = [row for idx, row in sorted(results, key=lambda x: x[0])]
label_df = pd.DataFrame(ordered_results, index=df.index)

# Merge back with original df in same order
output_df = pd.concat([df, label_df], axis=1)

output_df.to_csv(output_csv_path, index=False)
print(f"\nUpdated CSV saved to: {output_csv_path}")
