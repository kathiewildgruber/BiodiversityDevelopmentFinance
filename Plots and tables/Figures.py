from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Tuple

import math
import os

import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pycountry
import seaborn as sns
import statsmodels.api as sm
import warnings

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FormatStrFormatter
from numpy.polynomial import Polynomial
from plotly.subplots import make_subplots


# ------------------------------------------------------------------------------
# SET UP AND GENERAL
# ------------------------------------------------------------------------------

# Load CSV with explicit dtypes and proper memory handling
dtype_spec = {
    'Year': 'float64',
    'USD_Disbursement': 'float64'
}

df = pd.read_csv(
    # download file from google drive (link in readme) and add filepath here
    sep=';',
    dtype=dtype_spec,
    low_memory=False
)

print("Columns in the dataset:", df.columns.tolist())

# Ensure Year is numeric and within the desired range
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df[df['Year'].between(2000, 2023)]

# Add a Count column for project counting
df['Count'] = 1

# === REPLACE NEGATIVE USD_DISBURSEMENT VALUES ===

# Replace negative USD_Disbursements with 0 and report stats
negatives_df = df[df['USD_Disbursement'] < 0].copy()
num_negatives = len(negatives_df)
sum_negatives = negatives_df['USD_Disbursement'].sum()


# Replace the values
df.loc[df['USD_Disbursement'] < 0, 'USD_Disbursement'] = 0

# Report summary
# === CONVERT DATASET TO USD BIO (INSTEAD OF USD MIO) ===

df['USD_Disbursement'] = df['USD_Disbursement'] / 1000


# === FIX RECIPIENT NAMES FOR ISO3 CONVERSION ===

recipient_name_fixes = {
    "Saint Helena": "Saint Helena, Ascension and Tristan da Cunha",
    "China (People's Republic of)": "China",
    "West Bank and Gaza Strip": "Palestine, State of",
    "Micronesia": "Micronesia, Federated States of",
    "Democratic Republic of the Congo": "Congo, The Democratic Republic of the"
}

df['RecipientName'] = df['RecipientName'].replace(recipient_name_fixes)


# === CREATE OUTPUT FOLDER ===
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

# === DEFINE MULTI CLASSES ===

act_cols = ['Act_Pollut', 'Act_Invasiv', 'Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt',
            'Act_OthMgmt', 'Act_Protect', 'Act_Resto', 'Act_Undef']
act_merged_cols = ['Act_Pollut', 'Act_Invasiv', 'Act_SustResMgmt', 'Act_Protect_Resto', 'Act_Undef']
impl_cols = ['Impl_Regul', 'Impl_Know', 'Impl_Infra', 'Impl_Undef']
eco_cols = ['Eco_CropGrass', 'Eco_Forest', 'Eco_SeaWater', 'Eco_UrbInd', 'Eco_Undef']

# === DEFINE MULTI CLASS LABELS WITH LINEBREAKS ===

act_labels = {
    'Act_Pollut': 'Pollution\ncontrol',
    'Act_Invasiv': 'Invasive\nspecies\nmanagement',
    'Act_Agri': 'Sustainable\nagriculture',
    'Act_ForestMgmt': 'Sustainable\nforest\nmanagement',
    'Act_Fish': 'Sustainable\nfishery',
    'Act_WaterMgmt': 'Sustainable\nwater\nmanagement',
    'Act_OthMgmt': 'Sustainable\nmgmt. of other\nnatural\nresources',
    'Act_Protect': 'Protection\nand\nconservation',
    'Act_Resto': 'Restoration',
    'Act_Undef': 'Undefined'
}

act_merged_labels = {
    'Act_Pollut': 'Pollution\ncontrol',
    'Act_Invasiv': 'Invasive\nspecies\nmanagement',
    'Act_SustResMgmt': 'Sustainable\nresource\nmanagement',
    'Act_Protect_Resto': 'Protection,\nconservation\nand restoration',
    'Act_Undef': 'Undefined'
}

impl_labels = {
    'Impl_Regul': 'Policy,\nregulation\nand governance',
    'Impl_Know': 'Awareness\nand\nknowledge',
    'Impl_Infra': 'Infrastructure',
    'Impl_Undef': 'Undefined'
}

eco_labels = {
    'Eco_CropGrass': 'Crop-, range-,\ngrass-, and\narid land',
    'Eco_Forest': 'Forest',
    'Eco_SeaWater': 'Marine and\nfreshwater',
    'Eco_UrbInd': 'Urban\nand\nindustrial',
    'Eco_Undef': 'Undefined'
}

# === DEFINE MULTI CLASS LABELS WITHOUT LINE BREAKS ===

act_labels_nobreak = {
    'Act_Pollut': 'Pollution control',
    'Act_Invasiv': 'Invasive species management',
    'Act_Agri': 'Sustainable agriculture',
    'Act_ForestMgmt': 'Sustainable forest management',
    'Act_Fish': 'Sustainable fishery',
    'Act_WaterMgmt': 'Sustainable water management',
    'Act_OthMgmt': 'Sustainable mgmt. of other natural resources',
    'Act_Protect': 'Protection and conservation',
    'Act_Resto': 'Restoration',
    'Act_Undef': 'Undefined'
}

act_merged_labels_nobreak = {
    'Act_Pollut': 'Pollution control',
    'Act_Invasiv': 'Invasive species management',
    'Act_SustResMgmt': 'Sustainable resource management',
    'Act_Protect_Resto': 'Protection, conservation and restoration',
    'Act_Undef': 'Undefined'
}

impl_labels_nobreak = {
    'Impl_Regul': 'Policy, regulation and governance',
    'Impl_Know': 'Awareness and knowledge',
    'Impl_Infra': 'Infrastructure',
    'Impl_Undef': 'Undefined'
}

eco_labels_nobreak = {
    'Eco_CropGrass': 'Crop-, range-, grass-, and arid land',
    'Eco_Forest': 'Forest',
    'Eco_SeaWater': 'Marine and freshwater',
    'Eco_UrbInd': 'Urban and industrial',
    'Eco_Undef': 'Undefined'
}

# Hide all pandas FutureWarnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

warnings.filterwarnings(
    "ignore",
    message=".*Downcasting object dtype arrays on .fillna.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*array concatenation with empty entries is deprecated.*",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*not compatible with tight_layout.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*DataFrameGroupBy.apply operated on the grouping columns.*",
    category=DeprecationWarning,
)

# ------------------------------------------------------------------------------
# YEAR CONFIG
# ------------------------------------------------------------------------------

#Change according to which period should be analyzed
years_of_interest = [2018, 2019, 2020, 2021, 2022, 2023]
#years_of_interest = [2012, 2013, 2014, 2015, 2016, 2017]
#years_of_interest = [2006, 2007, 2008, 2009, 2010, 2011]
#years_of_interest = [2000, 2001, 2002, 2003, 2004, 2005]

start_year_of_interest = min(years_of_interest)
end_year_of_interest   = max(years_of_interest)


# ------------------------------------------------------------------------------
# PROJECT COUNT ANALYSIS PER SUBCATEGORY AND CO-OCCURRENCE
# ------------------------------------------------------------------------------

# Define relevant columns
all_subclasses = act_cols + impl_cols + eco_cols

# A) Single-subclass project count per year
single_subclass_counts = {}
for col in all_subclasses:
    yearly_counts = df[df[col] == 1].groupby('Year')['Count'].sum().reindex(years_of_interest, fill_value=0)
    single_subclass_counts[col] = yearly_counts

# B) Pairwise subclass co-occurrence counts per year
pairwise_counts = {}
for col1, col2 in combinations(all_subclasses, 2):
    yearly_counts = df[(df[col1] == 1) & (df[col2] == 1)].groupby('Year')['Count'].sum().reindex(years_of_interest, fill_value=0)
    pairwise_counts[(col1, col2)] = yearly_counts
    #print(f"\nPairwise counts for ({col1}, {col2}):\n{yearly_counts}")

# === Compute yearly averages (years of interest) ===

# Average for single-subclass
average_single_counts = {col: counts.mean() for col, counts in single_subclass_counts.items()}

# Average for pairwise subclass combinations
average_pairwise_counts = {pair: counts.mean() for pair, counts in pairwise_counts.items()}

# Compute grouped disbursements
def get_grouped_disb(top_df):
    act = calc_group_split(top_df, act_cols, 'USD_Disbursement')
    impl = calc_group_split(top_df, impl_cols, 'USD_Disbursement')
    eco = calc_group_split(top_df, eco_cols, 'USD_Disbursement')
    return act, impl, eco

# Helper function to compute proportionally distributed data
def calc_group_split(df, group_cols, value_col, filter_biodiv=False):
    group_data = {col: [] for col in group_cols}

    if filter_biodiv:
        df = df[df['Biodiversity'].isin([1, 2])]

    for _, row in df.iterrows():
        active = [col for col in group_cols if row.get(col) == 1]
        if not active:
            continue
        share = row[value_col] / len(active)
        for col in active:
            group_data[col].append((row['Year'], share))

    result = pd.DataFrame()
    for col in group_cols:
        temp = pd.DataFrame(group_data[col], columns=['Year', value_col])
        grouped = temp.groupby('Year').sum().rename(columns={value_col: col})
        result = pd.concat([result, grouped], axis=1)

    return result.fillna(0).sort_index()


def merge_act_groups(df_split, kind='disb'):
    """
    Merge specific activity groups into act_merged logic.
    :param df_split: result of calc_group_split (either disbursement or count)
    :param kind: 'disb' or 'count' (only used to label output)
    :return: DataFrame with merged act categories
    """
    df_merged = pd.DataFrame(index=df_split.index)

    df_merged['Act_Pollut'] = df_split.get('Act_Pollut', 0)
    df_merged['Act_Invasiv'] = df_split.get('Act_Invasiv', 0)

    df_merged['Act_SustResMgmt'] = df_split[['Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt', 'Act_OthMgmt']].sum(axis=1)
    df_merged['Act_Protect_Resto'] = df_split[['Act_Protect', 'Act_Resto']].sum(axis=1)

    df_merged['Act_Undef'] = df_split.get('Act_Undef', 0)

    return df_merged


# Combine into one DataFrame
def combine_for_plot(year, *dfs_and_labels):
    rows = []
    for df, label in dfs_and_labels:
        if year not in df.index:
            raise ValueError(f"Year {year} not found in DataFrame index.")
        temp = df.loc[[year]].copy()
        temp['Group'] = label
        temp.set_index('Group', inplace=True)
        rows.append(temp)
    return pd.concat(rows)

def summarize(group):
    disb_series = group['USD_Disbursement']
    disb_series = disb_series[disb_series > 0]

    return pd.DataFrame({
        'Count': group['Count'].sum(),
        'Mean_Disb': disb_series.mean(),
        'Median_Disb': disb_series.median(),
        'Max_Disb': disb_series.max(),
        'Total_Disb': disb_series.sum()
    }, index=[group['Year'].iloc[0]])

def compute_average_totals(df, subclass_list, years_of_interest):
    """
    Computes average yearly disbursements for each subclass in a given dimension,
    proportionally splitting disbursements among active subclasses *in that dimension only*.
    """
    totals = {}
    df = df[(df['USD_Disbursement'] > 0)].copy()
    for subclass in subclass_list:
        yearly_sums = []
        for year in years_of_interest:
            df_year = df[df['Year'] == year]
            active_mask = df_year[subclass] == 1
            if active_mask.sum() == 0:
                yearly_sums.append(0)
                continue
            # Number of active columns per project *in this dimension*
            per_row_active = df_year[subclass_list].sum(axis=1)
            share = (df_year.loc[active_mask, 'USD_Disbursement'] / per_row_active[active_mask])
            yearly_sums.append(share.sum())
        avg = np.mean(yearly_sums)
        totals[subclass] = avg
    return totals

# === DEFINE BIODIV GROUPS ===

# 1) All ODA Projects
all_stats = df.groupby('Year').apply(summarize).droplevel(1)
all_stats.columns = [f'All_{col}' for col in all_stats.columns]

# 2) Biodiversity Rio Marker Projects (1 or 2)
biodiv_rio_df = df[df['Biodiversity'].isin([1, 2])]
biodiv_stats = biodiv_rio_df.groupby('Year').apply(summarize).droplevel(1)
biodiv_stats.columns = [f'BiodivRio_{col}' for col in biodiv_stats.columns]

# 3) BiodivBERT & LLM Filtered
bert_llm_df = df[(df['binary_label_biodiversity_impact'] == 1) & (df['No_Biodiv'] != 1)]
bert_stats = bert_llm_df.groupby('Year').apply(summarize).droplevel(1)
bert_stats.columns = [f'BiodivBERT_LLM_{col}' for col in bert_stats.columns]

# === Customized Orders and Color Palettes ===

# Targeted Action
act_order = ['Act_Protect','Act_Resto', 'Act_Agri','Act_ForestMgmt', 'Act_Fish','Act_WaterMgmt','Act_OthMgmt',
             'Act_Invasiv', 'Act_Pollut', 'Act_Undef']
act_colors = ['#e2d0ea','#c39acd','#b581bf','#a467b0','#944ea2','#843894','#773286','#6b2d78','#512e5f', 'lightgrey']

act_merged_order = ['Act_Protect_Resto', 'Act_SustResMgmt','Act_Invasiv', 'Act_Pollut', 'Act_Undef']
act_merged_colors = ['#e2d0ea','#944ea2','#6b2d78','#512e5f','lightgrey']

# Implementation Tools
impl_order = [ 'Impl_Regul', 'Impl_Know', 'Impl_Infra', 'Impl_Undef']
impl_colors = ['#d0f4aa', '#d0d97d', '#aab63d', 'lightgrey']

# Ecosystem
eco_order = ['Eco_CropGrass','Eco_Forest','Eco_SeaWater','Eco_UrbInd','Eco_Undef']
eco_colors = ['#b1f1dc', '#84dbc0', '#6bc1a4', '#55a88f', 'lightgrey']

# ------------------------------------------------------------------------------
# FIGURES
# ------------------------------------------------------------------------------

# Filter to bert_llm subset with positive disbursements only
df_filtered = df[(df['binary_label_biodiversity_impact'] == 1) & (df['No_Biodiv'] != 1) & (df['USD_Disbursement'] > 0)].copy()

# Calculate grouped disbursements for Action, Implementation, Ecosystem
act_disb, impl_disb, eco_disb= get_grouped_disb(df_filtered)

# Filter by years of interest
act_disb_filtered = act_disb.loc[years_of_interest]
impl_disb_filtered = impl_disb.loc[years_of_interest]
eco_disb_filtered = eco_disb.loc[years_of_interest]

# Build matrix DataFrame with subclasses as rows and years as columns
matrix_data = []

for subclass in act_cols:
    if subclass in act_disb_filtered.columns:
        matrix_data.append({
            'Subclass': subclass,
            'Dimension': 'Goal',
            **{year: act_disb_filtered.loc[year, subclass] for year in years_of_interest}
        })

for subclass in impl_cols:
    if subclass in impl_disb_filtered.columns:
        matrix_data.append({
            'Subclass': subclass,
            'Dimension': 'Instrument',
            **{year: impl_disb_filtered.loc[year, subclass] for year in years_of_interest}
        })

for subclass in eco_cols:
    if subclass in eco_disb_filtered.columns:
        matrix_data.append({
            'Subclass': subclass,
            'Dimension': 'Ecosystem',
            **{year: eco_disb_filtered.loc[year, subclass] for year in years_of_interest}
        })

matrix_df = pd.DataFrame(matrix_data)

# Recalculate accurate yearly average totals (using shared logic)
df_biodiv = df[(df['binary_label_biodiversity_impact'] == 1) & (df['No_Biodiv'] != 1) & (df['USD_Disbursement'] > 0)].copy()

act_total_avg = compute_average_totals(df_biodiv, act_cols, years_of_interest)
impl_total_avg = compute_average_totals(df_biodiv, impl_cols, years_of_interest)
eco_total_avg = compute_average_totals(df_biodiv, eco_cols, years_of_interest)


totals_data = []

for subclass, value in act_total_avg.items():
    totals_data.append({
        'Subclass': subclass,
        'Dimension': 'Goal',
        'Average_Annual': value,
        **{year: np.nan for year in years_of_interest}
    })

for subclass, value in impl_total_avg.items():
    totals_data.append({
        'Subclass': subclass,
        'Dimension': 'Instrument',
        'Average_Annual': value,
        **{year: np.nan for year in years_of_interest}
    })

for subclass, value in eco_total_avg.items():
    totals_data.append({
        'Subclass': subclass,
        'Dimension': 'Ecosystem',
        'Average_Annual': value,
        **{year: np.nan for year in years_of_interest}
    })


final_matrix = pd.concat([matrix_df, pd.DataFrame(totals_data)], ignore_index=True)

# Add average column across years
final_matrix['Average_Annual'] = final_matrix[years_of_interest].mean(axis=1)

# Reorder columns
column_order = ['Subclass', 'Dimension', 'Average_Annual'] + years_of_interest
final_matrix = final_matrix[column_order]


# ------------------------------------------------------------------------------
# SI: INCOME GROUP DIMENSIONS
# ------------------------------------------------------------------------------

# Define income groups (column name: IncomegroupName)
income_groups = ["L", "LM", "UM", "H", ""]
# Convert missing income groups to empty string for plotting
df['WB_Incomegroup'] = df['WB_Incomegroup'].fillna("")


# Filter main analysis subset (same logic as df_filtered)
df_income = df[
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1) &
    (df['USD_Disbursement'] > 0) &
    (df['Year'].isin(years_of_interest)) &
    (df['WB_Incomegroup'].isin(income_groups))
].copy()

# Calculate annual average disbursement (years of interes) for rows with missing WB_Incomegroup
df_missing_income = df[
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1) &
    (df['USD_Disbursement'] > 0) &
    (df['Year'].isin(years_of_interest)) &
    (df['WB_Incomegroup'].isna())
].copy()

# Function: compute proportional averages
def compute_avg_disbursements_by_income(df, subclass_list):
    results = {group: {sub: 0 for sub in subclass_list} for group in income_groups}

    for year in years_of_interest:
        df_year = df[df['Year'] == year].copy()
        df_year['active'] = df_year[subclass_list].sum(axis=1)
        df_year = df_year[df_year['active'] > 0]
        df_year['split'] = df_year['USD_Disbursement'] / df_year['active']

        for subclass in subclass_list:
            df_sub = df_year[df_year[subclass] == 1]
            grouped = df_sub.groupby('WB_Incomegroup')['split'].sum()

            for group, val in grouped.items():
                if group in results:
                    results[group][subclass] += val / len(years_of_interest)

    return results


# Compute data
act_income_data = compute_avg_disbursements_by_income(df_income, act_order)
impl_income_data = compute_avg_disbursements_by_income(df_income, impl_order)
eco_income_data = compute_avg_disbursements_by_income(df_income, eco_order)

# Custom display names for income groups
income_label_map = {
    "L": "Low\nIncome",
    "LM": "Lower-\nMiddle\nIncome",
    "UM": "Upper-\nMiddle\nIncome",
    "H": "High\nIncome",
    "": "Un-\nspecified"
}

# Plotting Function: 3-panel layout with 100% bars + totals
def plot_income_stacked_bars(act_data, impl_data, eco_data):
    fig, axs = plt.subplots(1, 3, figsize=(18, 14), sharex=True)

    for ax, data_dict, labels_dict, title, colors, col_order in zip(
            axs,
            [act_data, impl_data, eco_data],
            [act_labels_nobreak, impl_labels_nobreak, eco_labels_nobreak],
            ['Goal', 'Instrument', 'Ecosystem'],
            [act_colors, impl_colors, eco_colors],
            [act_order, impl_order, eco_order]
    ):
        df_plot = pd.DataFrame({group: data_dict[group] for group in income_groups})
        df_plot = df_plot.loc[col_order]
        df_plot.index = [labels_dict.get(row, row) for row in col_order]

        abs_totals = df_plot.sum(axis=0)
        df_percent = df_plot.div(abs_totals, axis=1).fillna(0) * 100

        bottom = np.zeros(len(income_groups))
        bar_handles = []

        for idx, label in enumerate(df_plot.index):
            values = df_percent.loc[label].values
            bars = ax.bar(income_groups, values, bottom=bottom, label=label, color=colors[idx])
            bottom += values
            bar_handles.append(bars[0])

            for bar, pct in zip(bars, values):
                if pct > 5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + pct / 2,
                        f"{pct:.0f}%",
                        ha='center',
                        va='center',
                        fontsize=12,
                        color='white'
                    )

        for i, group in enumerate(income_groups):
            val = abs_totals.get(group, 0)
            ax.text(
                i, 103,
                f"${val:.1f}B",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(range(len(income_groups)))
        ax.set_xticklabels([income_label_map.get(g, g) for g in income_groups], rotation=0, ha='center', fontsize=14)
        ax.set_ylim(0, 117)
        ax.set_ylabel("Share of Total (%)", fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Clean, compact, top-aligned legend below each subplot
        ax.legend(
            handles=bar_handles,
            labels=df_plot.index.tolist(),
            loc='upper left',
            bbox_to_anchor=(0, -0.2),  # << raised legend closer to chart
            ncol=1,
            fontsize=14,
            frameon=True,
            fancybox=True
        )

    plt.tight_layout(rect=[0, 0.25, 1, 1])  # reserves space but trims excess
    plt.subplots_adjust(wspace=0.3)  # Optional fine-tuning for horizontal spacing
    plt.savefig("outputs/SI_Income_Groups_Dimensions.png", dpi=300)
    plt.savefig("outputs/SI_Income_Groups_Dimensions.pdf", dpi=300)
    plt.show()


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# 1. Recreate merged subclass column before filtering
df['Act_SustResMgmt'] = df[['Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt', 'Act_OthMgmt']].max(axis=1)
df['Act_Protect_Resto'] = df[['Act_Protect', 'Act_Resto']].max(axis=1)

# 2. Define income groups
income_groups = ["L", "LM", "UM", "H",""]

# Convert missing income groups to empty string for plotting
df['WB_Incomegroup'] = df['WB_Incomegroup'].fillna("")

# 3. Filter main analysis subset
df_income = df[
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1) &
    (df['USD_Disbursement'] > 0) &
    (df['Year'].isin(years_of_interest)) &
    (df['WB_Incomegroup'].isin(income_groups))
].copy()

# 4. Log average disbursement for rows with missing WB_Incomegroup
df_missing_income = df[
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1) &
    (df['USD_Disbursement'] > 0) &
    (df['Year'].isin(years_of_interest)) &
    (df['WB_Incomegroup'].isna())
].copy()

# 5. Use merged act list
act_cols_merged = ['Act_Protect_Resto', 'Act_SustResMgmt', 'Act_Invasiv', 'Act_Pollut', 'Act_Undef']

# 6. Compute updated data
def compute_avg_disbursements_by_income(df, subclass_list):
    results = {group: {sub: 0 for sub in subclass_list} for group in income_groups}

    for year in years_of_interest:
        df_year = df[df['Year'] == year].copy()
        df_year['active'] = df_year[subclass_list].sum(axis=1)
        df_year = df_year[df_year['active'] > 0]
        df_year['split'] = df_year['USD_Disbursement'] / df_year['active']

        for subclass in subclass_list:
            df_sub = df_year[df_year[subclass] == 1]
            grouped = df_sub.groupby('WB_Incomegroup')['split'].sum()

            for group, val in grouped.items():
                if group in results:
                    results[group][subclass] += val / len(years_of_interest)

    return results

# 7. Compute income data with merged action categories
act_income_data_merged = compute_avg_disbursements_by_income(df_income, act_cols_merged)
impl_income_data = compute_avg_disbursements_by_income(df_income, impl_order)
eco_income_data = compute_avg_disbursements_by_income(df_income, eco_order)

# 8. Income group display names
income_label_map = {
    "L": "Low\nIncome",
    "LM": "Lower-\nMiddle\nIncome",
    "UM": "Upper-\nMiddle\nIncome",
    "H": "High\nIncome",
    "": "Un-\nspecified"
}

# 9. Plotting function for merged category version
def plot_income_stacked_bars_merged(act_data, impl_data, eco_data):
    fig, axs = plt.subplots(1, 3, figsize=(18, 12), sharex=True)

    for ax, data_dict, labels_dict, title, colors, col_order in zip(
            axs,
            [act_data, impl_data, eco_data],
            [act_merged_labels_nobreak, impl_labels_nobreak, eco_labels_nobreak],
            ['Goal', 'Instrument', 'Ecosystem'],
            [act_merged_colors, impl_colors, eco_colors],
            [act_merged_order, impl_order, eco_order]
    ):
        df_plot = pd.DataFrame({group: data_dict[group] for group in income_groups})
        df_plot = df_plot.loc[col_order]
        df_plot.index = [labels_dict.get(row, row) for row in col_order]

        abs_totals = df_plot.sum(axis=0)
        df_percent = df_plot.div(abs_totals, axis=1).fillna(0) * 100

        bottom = np.zeros(len(income_groups))
        bar_handles = []

        for idx, label in enumerate(df_plot.index):
            values = df_percent.loc[label].values
            bars = ax.bar(income_groups, values, bottom=bottom, label=label, color=colors[idx])
            bottom += values
            bar_handles.append(bars[0])

            for bar, pct in zip(bars, values):
                if pct > 5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + pct / 2,
                        f"{pct:.0f}%",
                        ha='center',
                        va='center',
                        fontsize=12,
                        color='white'
                    )

        for i, group in enumerate(income_groups):
            val = abs_totals.get(group, 0)
            ax.text(
                i, 103,
                f"${val:.1f}B",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(range(len(income_groups)))
        ax.set_xticklabels([income_label_map.get(g, g) for g in income_groups], rotation=0, ha='center', fontsize=14)
        ax.set_ylim(0, 117)
        ax.set_ylabel("Share of Total (%)", fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Clean, compact, top-aligned legend below each subplot
        ax.legend(
            handles=bar_handles,
            labels=df_plot.index.tolist(),
            loc='upper left',
            bbox_to_anchor=(0, -0.15),
            ncol=1,
            fontsize=14,
            frameon=True,
            fancybox=True
        )

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# 1. Recreate merged subclass column before filtering
df['Act_SustResMgmt'] = df[['Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt', 'Act_OthMgmt']].max(axis=1)
df['Act_Protect_Resto'] = df[['Act_Protect', 'Act_Resto']].max(axis=1)

# 2. Define income groups
income_groups = ["L", "LM", "UM", "H",""]

df['WB_Incomegroup'] = df['WB_Incomegroup'].fillna("")


# 3. Filter main analysis subset
df_income = df[
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1) &
    (df['USD_Disbursement'] > 0) &
    (df['Year'].isin(years_of_interest)) &
    (df['WB_Incomegroup'].isin(income_groups))
].copy()

# 4. Log average disbursement for rows with missing WB_Incomegroup
df_missing_income = df[
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1) &
    (df['USD_Disbursement'] > 0) &
    (df['Year'].isin(years_of_interest)) &
    (df['WB_Incomegroup'].isna())
].copy()

# Combine specified and missing income group data
df_income_for_total = pd.concat([df_income, df_missing_income], axis=0)

# Total disbursement by income group (years of interes) in billions
disbursement_per_income = (
    df_income_for_total.groupby('WB_Incomegroup')['USD_Disbursement'].sum()
    .reindex(income_groups, fill_value=0)
)

# 5. Use merged act list
act_cols_merged = ['Act_Protect_Resto', 'Act_SustResMgmt', 'Act_Invasiv', 'Act_Pollut', 'Act_Undef']

# 6. Compute income data with merged action categories
act_income_data_merged = compute_avg_disbursements_by_income(df_income, act_cols_merged)
impl_income_data = compute_avg_disbursements_by_income(df_income, impl_order)
eco_income_data = compute_avg_disbursements_by_income(df_income, eco_order)

# 7. Compute total average disbursement (per income group)

total_disbursement_dict = {
    group: sum(act_income_data_merged[group].values())
    for group in income_groups
}


# 8. Income group display names
income_label_map = {
    "L": "Low\nIncome",
    "LM": "Lower-\nMiddle\nIncome",
    "UM": "Upper-\nMiddle\nIncome",
    "H": "High\nIncome",
    "": "Un-\nspecified"
}

# 9. Plotting function for merged category version
def plot_income_stacked_bars_merged(act_data, impl_data, eco_data, total_disbursement_dict):


    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(1, 4, width_ratios=[0.3, 1, 1, 1])
    axs = [plt.subplot(gs[i]) for i in range(4)]

    # GREYSCALE COLORS for Total Disbursement (dark to light)
    grey_colors = ['#3a3a3a', '#5a5a5a', '#7a7a7a', '#a0a0a0', '#d5d5d5']

    # Values in income group order
    vals = [total_disbursement_dict.get(g, 0) for g in income_groups]

    # ----- LEFTMOST PLOT: STACKED BAR for TOTAL DISBURSEMENT (PERCENTAGES) -----
    total_val = sum(vals)
    bottom = 0
    bar_handles = []
    bar_width = 0.6

    # Convert values to percentages
    vals_pct = [(v / total_val) * 100 if total_val > 0 else 0 for v in vals]

    for i, (group, pct, color) in enumerate(zip(income_groups, vals_pct, grey_colors)):
        bar = axs[0].bar(
            x=[0],
            height=[pct],
            bottom=bottom,
            width=bar_width,
            color=color,
            label=income_label_map[group]
        )
        if pct > 0:
            axs[0].text(
                x=0,
                y=bottom + pct / 2,
                s=f"{pct:.0f}%",
                ha='center',
                va='center',
                fontsize=12,
                color='white' if i < 3 else 'black'
            )
        bar_handles.append(bar[0])
        bottom += pct

    # === NEW: Absolute total ($B) above the bar, within the plot ===
    axs[0].text(
        x=0,
        y=103,
        s=f"${total_val:.1f}B",
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )

    # Tidy up plot
    axs[0].set_xlim(-0.6, 0.6)
    axs[0].set_ylim(0, 115)
    axs[0].set_title("Total", fontsize=16, fontweight='bold', pad=15)
    axs[0].set_xticks([0])
    axs[0].set_xticklabels([f"Average annual\nDisbursement\n{start_year_of_interest}–{end_year_of_interest}"], fontsize=14)
    axs[0].set_ylabel("Share of Disbursement (%)", fontsize=14)
    axs[0].tick_params(axis='y', labelsize=12)

    axs[0].legend(
        handles=bar_handles,
        labels=[income_label_map[g] for g in income_groups],
        loc='upper left',
        bbox_to_anchor=(0, -0.2),
        ncol=1,
        fontsize=14,
        frameon=True,
        fancybox=True
    )

    # ----- REMAINING 3 PLOTS -----
    for ax, data_dict, labels_dict, title, colors, col_order in zip(
            axs[1:],  # Shifted to skip the 0th subplot
            [act_data, impl_data, eco_data],
            [act_merged_labels_nobreak, impl_labels_nobreak, eco_labels_nobreak],
            ['Goal', 'Instrument', 'Ecosystem'],
            [act_merged_colors, impl_colors, eco_colors],
            [act_merged_order, impl_order, eco_order]
    ):
        df_plot = pd.DataFrame({group: data_dict[group] for group in income_groups})
        df_plot = df_plot.loc[col_order]
        df_plot.index = [labels_dict.get(row, row) for row in col_order]

        abs_totals = df_plot.sum(axis=0)
        df_percent = df_plot.div(abs_totals, axis=1).fillna(0) * 100

        bottom = np.zeros(len(income_groups))
        bar_handles = []

        for idx, label in enumerate(df_plot.index):
            values = df_percent.loc[label].values
            bars = ax.bar(income_groups, values, bottom=bottom, label=label, color=colors[idx])
            bottom += values
            bar_handles.append(bars[0])

            for bar, pct in zip(bars, values):
                if pct > 5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + pct / 2,
                        f"{pct:.0f}%",
                        ha='center',
                        va='center',
                        fontsize=12,
                        color='white'
                    )

        for i, group in enumerate(income_groups):
            val = abs_totals.get(group, 0)
            ax.text(
                i, 103,
                f"${val:.1f}B",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(range(len(income_groups)))
        ax.set_xticklabels([income_label_map.get(g, g) for g in income_groups], rotation=0, ha='center', fontsize=14)
        ax.set_ylim(0, 115)
        ax.tick_params(axis='y', labelsize=12, labelcolor='white')

        ax.legend(
            handles=bar_handles,
            labels=df_plot.index.tolist(),
            loc='upper left',
            bbox_to_anchor=(0, -0.2),
            ncol=1,
            fontsize=14,
            frameon=True,
            fancybox=True
        )

disbursement_avg_per_income = disbursement_per_income / len(years_of_interest)


#-----------------------------------------
# CONFIGURATION
#-----------------------------------------


# 1) Setup & Safe Fallbacks
income_groups = ["L", "LM", "UM", "H", ""]

# Display labels for x-axis
income_label_map = {
    "L": "Low\nIncome",
    "LM": "Lower-\nMiddle\nIncome",
    "UM": "Upper-\nMiddle\nIncome",
    "H": "High\nIncome",
    "": "Un-\nspecified",
}

country_col = "RecipientName"
if country_col not in df.columns:
    raise KeyError(f"Expected country column '{country_col}' not found in df.")
if "WB_Incomegroup" not in df.columns:
    raise KeyError("Expected column 'WB_Incomegroup' not found in df.")

# Normalize WB_Incomegroup values to codes L/LM/UM/H
def _map_income_code(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s_low = s.lower()
    if s in {"L", "LM", "UM", "H", ""}:
        return s
    if s_low.startswith("low ") and "middle" not in s_low:
        return "L"
    if s_low.startswith("lower") or "lower middle" in s_low:
        return "LM"
    if s_low.startswith("upper") or "upper middle" in s_low:
        return "UM"
    if s_low.startswith("high"):
        return "H"
    return ""

df["WB_Incomegroup"] = df["WB_Incomegroup"].apply(_map_income_code).fillna("")


# 2) Filter to main analysis subset
df_income = df[
    (df["binary_label_biodiversity_impact"] == 1)
    & (df["No_Biodiv"] != 1)
    & (df["USD_Disbursement"] > 0)
    & (df["Year"].isin(years_of_interest))
    & (df["WB_Incomegroup"].isin(income_groups))
].copy()

# Compute average yearly disbursement (USD billion) per income group
avg_disb_per_income = (
    df_income.groupby("WB_Incomegroup")["USD_Disbursement"].sum() / len(years_of_interest)
).reindex(income_groups, fill_value=0)


# 3) Core computation
def _mode_value(series):
    counts = Counter(series.dropna().tolist())
    if not counts:
        return None
    max_count = max(counts.values())
    candidates = sorted([k for k, v in counts.items() if v == max_count])
    return candidates[0]

def compute_country_normalized_shares(df_in: pd.DataFrame, subclass_list):
    bucket = {g: [] for g in income_groups}

    for country, gdf in df_in.groupby(country_col):
        inc = _mode_value(gdf["WB_Incomegroup"])
        if inc not in bucket:
            continue

        split = calc_group_split(gdf, subclass_list, "USD_Disbursement", filter_biodiv=False)
        split = split.loc[split.index.intersection(pd.Index(years_of_interest))]
        if split.empty:
            continue

        avg_abs = split.mean(axis=0)
        total = avg_abs.sum()
        if total <= 0:
            continue

        pct = (avg_abs / total) * 100.0
        pct = pct.reindex(subclass_list, fill_value=0)
        bucket[inc].append(pct)

    results = {}
    for inc, series_list in bucket.items():
        if series_list:
            mat = pd.DataFrame(series_list)
            results[inc] = mat.mean(axis=0)
        else:
            results[inc] = pd.Series(0.0, index=subclass_list)
    return results

# Compute for each dimension
act_rel = compute_country_normalized_shares(df_income, act_order)
impl_rel = compute_country_normalized_shares(df_income, impl_order)
eco_rel = compute_country_normalized_shares(df_income, eco_order)

# 4) Plotting (3-panel stacked % bars)
def _plot_three_panel_relative(act_rel, impl_rel, eco_rel, avg_disb_per_income):
    fig, axs = plt.subplots(1, 3, figsize=(18, 12), sharex=False)

    panels = [
        (axs[0], act_rel, act_order, act_labels_nobreak, act_colors, "Goal"),
        (axs[1], impl_rel, impl_order, impl_labels_nobreak, impl_colors, "Instrument"),
        (axs[2], eco_rel, eco_order, eco_labels_nobreak, eco_colors, "Ecosystem"),
    ]

    for idx, (ax, data_rel, order, labels_map, colors, title) in enumerate(panels):
        df_plot = pd.DataFrame({inc: data_rel.get(inc, pd.Series(0, index=order)) for inc in income_groups})
        df_plot = df_plot.loc[order].fillna(0)
        df_plot.index = [labels_map.get(c, c) for c in order]
        df_plot = df_plot.apply(lambda col: (col / max(col.sum(), 1e-12)) * 100, axis=0)

        bottoms = np.zeros(len(income_groups))
        handles = []

        for i, cls in enumerate(df_plot.index):
            vals = df_plot.loc[cls].values
            bars = ax.bar(range(len(income_groups)), vals, bottom=bottoms, label=cls, color=colors[i])
            handles.append(bars[0])

            # Add inside-bar percentage labels only
            for bar, pct in zip(bars, vals):
                if pct >= 7:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + pct / 2,
                        f"{pct:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="white",
                    )

            bottoms += vals

        # --- REMOVED TOTALS ABOVE EACH BAR ---
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticks(range(len(income_groups)))
        ax.set_xticklabels([income_label_map.get(g, g) for g in income_groups], fontsize=13)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="y", labelsize=12)

        # Only left-most subplot has y-axis label
        if idx == 0:
            ax.set_ylabel("Average national funding allocation (%)", fontsize=12)
        else:
            ax.set_ylabel("")

        ax.legend(
            handles=handles,
            labels=df_plot.index.tolist(),
            loc="upper left",
            bbox_to_anchor=(0, -0.22),
            ncol=1,
            fontsize=12,
            frameon=True,
            fancybox=True,
        )

    plt.tight_layout(rect=[0, 0.25, 1, 1])
    plt.subplots_adjust(wspace=0.3)
    plt.savefig("outputs/SI_Income_Group_RelativeShares.png", dpi=300)
    plt.savefig("outputs/SI_Income_Group_RelativeShares.pdf", dpi=300)
    plt.show()


# Call plotting function
plot_income_stacked_bars(act_income_data, impl_income_data, eco_income_data)

# 5) Run plot
_plot_three_panel_relative(act_rel, impl_rel, eco_rel, avg_disb_per_income)


# ===========================================
# SI: INCOME GROUP AND FINANCE FLOW TYPE
# ===========================================

# -----------------------
# Config / assumptions
# -----------------------
income_groups = ["L", "LM", "UM", "H", ""]  # "" = Unspecified
income_label_map_breaks = {
    "L": "Low\nIncome",
    "LM": "Lower-\nMiddle\nIncome",
    "UM": "Upper-\nMiddle\nIncome",
    "H": "High\nIncome",
    "": "Un-\nspecified",
}

flow_order = [
    "ODA Grants",
    "ODA Loans",
    "Other Official Flows (non Export Credit)",
    "Private Development Finance",
    "Equity Investment",
]

flow_color_map = {
    "ODA Grants": "#d9d9d9",
    "ODA Loans": "#bfbfbf",
    "Other Official Flows (non Export Credit)": "#999999",
    "Private Development Finance": "#595959",
    "Equity Investment": "#1a1a1a",
}

df["WB_Incomegroup"] = df["WB_Incomegroup"].fillna("").infer_objects(copy=False)

# -----------------------
# Base filter
# -----------------------
df_base = df[
    (df["binary_label_biodiversity_impact"] == 1) &
    (df["No_Biodiv"] != 1) &
    (df["USD_Disbursement"] > 0) &
    (df["Year"].isin(years_of_interest)) &
    (df["WB_Incomegroup"].isin(income_groups))
].copy()

if df_base.empty:
    raise ValueError("Filtered dataset is empty with the current filters.")

# -----------------------
# Computation
# -----------------------
def compute_avg_totals_by_income(df_in: pd.DataFrame):
    """Average annual total USD_Disbursement per income group across years_of_interest."""
    n_years = len(years_of_interest)
    totals = {g: 0 for g in income_groups}

    for year in years_of_interest:
        df_year = df_in[df_in["Year"] == year]
        yr_totals = df_year.groupby("WB_Incomegroup")["USD_Disbursement"].sum()
        for g, v in yr_totals.items():
            if g in totals:
                totals[g] += v / n_years
    return totals


def compute_avg_disbursements_by_income_flow(df_in: pd.DataFrame, main_flows: list):
    """
    Returns dict: {income_group: {flow: avg_annual_sum_over_years}}
    Only includes flows in main_flows; all other FlowName values are ignored.
    """
    n_years = len(years_of_interest)
    result = {g: {f: 0 for f in main_flows} for g in income_groups}

    d = df_in[df_in["FlowName"].notna()].copy()

    for year in years_of_interest:
        df_year = d[d["Year"] == year]
        sums = df_year.groupby(["WB_Incomegroup", "FlowName"])["USD_Disbursement"].sum()

        for (g, f), v in sums.items():
            if (g in result) and (f in result[g]):
                result[g][f] += v / n_years

    return result


income_totals = compute_avg_totals_by_income(df_base)
flow_income_data = compute_avg_disbursements_by_income_flow(df_base, flow_order)

pivot_overall = (
    pd.DataFrame({g: flow_income_data[g] for g in income_groups})
      .T
      .reindex(income_groups)
      .fillna(0)
)

# Ensure column order
pivot_overall = pivot_overall[[c for c in flow_order if c in pivot_overall.columns]]

# Percent shares within EACH income group using ONLY the displayed flows (same as your calc code)
pivot_percent = pivot_overall.div(pivot_overall.sum(axis=1), axis=0).fillna(0) * 100

# Totals displayed above bars: from income_totals (full df_base, matches dimensions)
abs_totals = pd.Series({g: income_totals.get(g, 0) for g in income_groups}).reindex(income_groups)

# -----------------------
# Plot
# -----------------------

out_dir = "./outputs"
os.makedirs(out_dir, exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=False)

# Left panel off
axs[0].axis("off")

# Right panel for legend
axs[2].axis("off")

# Middle panel is the chart
ax = axs[1]

x = np.arange(len(income_groups))
bar_width = 0.65

bottom = np.zeros(len(income_groups), dtype=float)
handles, labels = [], []

for i, flow in enumerate(flow_order):
    if flow not in pivot_percent.columns:
        continue

    heights = pivot_percent[flow].values
    color = flow_color_map.get(flow, "#cccccc")

    bars = ax.bar(
        x, heights, bottom=bottom, width=bar_width,
        color=color, label=flow
    )

    # Inside % labels
    for bar, hgt, y0 in zip(bars, heights, bottom):
        if hgt > 5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y0 + hgt / 2,
                f"{hgt:.0f}%",
                ha="center", va="center",
                fontsize=11,
                color="white" if i >= 2 else "black",
                fontweight="bold"
            )

    bottom += heights
    handles.append(bars[0])
    labels.append(flow)

# Totals above bars
for xi, g in zip(x, income_groups):
    total = abs_totals.loc[g]
    ax.text(
        xi, 106, f"${total:.1f}B",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold"
    )

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels([income_label_map_breaks[g] for g in income_groups], fontsize=13)
ax.set_ylim(0, 115)
ax.tick_params(axis="y", labelsize=12)
ax.set_ylabel(
    f"Annual average disbursement (USD bn and %, {start_year_of_interest}-{end_year_of_interest})",
    fontsize=12
)

# Legend inside rightmost panel
axs[2].legend(
    handles, labels, fontsize=12, frameon=True, fancybox=True,
    loc="center",
)

plt.tight_layout(rect=[0, 0, 1, 1])

# Save with requested name
plt.savefig(os.path.join(out_dir, "SI_Income_Group_and_Finance_Flow_Types.png"), dpi=300)
plt.savefig(os.path.join(out_dir, "SI_Income_Group_and_Finance_Flow_Types.pdf"), dpi=300)
plt.show()


#-------------------------------------------
# CONFIGURATION
#-------------------------------------------

# Create merged Act columns (idempotent; safe if they already exist)
df['Act_SustResMgmt'] = df[['Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt', 'Act_OthMgmt']].max(axis=1)
df['Act_Protect_Resto'] = df[['Act_Protect', 'Act_Resto']].max(axis=1)

# Default merged order (used if act_merged_order not provided upstream)
if 'act_merged_order' not in globals():
    act_merged_order = ['Act_Protect_Resto', 'Act_SustResMgmt', 'Act_Invasiv', 'Act_Pollut', 'Act_Undef']


def _titleize(key):
    return key.replace('_', ' ').replace('Act ', '').strip().title()

if 'act_merged_labels_nobreak' not in globals():
    act_merged_labels_nobreak = {}

    if 'act_labels_nobreak' in globals():
        # sensible defaults for merged buckets
        act_merged_labels_nobreak.update({
            'Act_Protect_Resto': act_labels_nobreak.get('Act_Protect_Resto', 'Protect/\nRestore'),
            'Act_SustResMgmt'  : act_labels_nobreak.get('Act_SustResMgmt', 'Sustainable\nMgmt'),
            'Act_Invasiv'      : act_labels_nobreak.get('Act_Invasiv', 'Invasive\nSpecies'),
            'Act_Pollut'       : act_labels_nobreak.get('Act_Pollut', 'Pollution\nControl'),
            'Act_Undef'        : act_labels_nobreak.get('Act_Undef', 'Unspecified')
        })
    else:
        act_merged_labels_nobreak = {
            'Act_Protect_Resto': 'Protect/\nRestore',
            'Act_SustResMgmt'  : 'Sustainable\nMgmt',
            'Act_Invasiv'      : 'Invasive\nSpecies',
            'Act_Pollut'       : 'Pollution\nControl',
            'Act_Undef'        : 'Unspecified'
        }


if 'act_merged_colors' not in globals():

    color_map = {}
    if 'act_order' in globals() and 'act_colors' in globals() and len(act_order) == len(act_colors):
        color_map = {k: v for k, v in zip(act_order, act_colors)}

    def _pick_color(key, fallbacks):

        for fb in fallbacks:
            if fb in color_map:
                return color_map[fb]
        return '#9e9e9e'
    act_merged_colors = [
        _pick_color('Act_Protect_Resto', ['Act_Protect', 'Act_Resto']),
        _pick_color('Act_SustResMgmt',   ['Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt', 'Act_OthMgmt']),
        _pick_color('Act_Invasiv',       ['Act_Invasiv']),
        _pick_color('Act_Pollut',        ['Act_Pollut']),
        _pick_color('Act_Undef',         ['Act_Undef']),
    ]


# 1) Setup & income groups
income_groups = ["L", "LM", "UM", "H", ""]
income_label_map = {
    "L": "Low\nIncome",
    "LM": "Lower-\nMiddle\nIncome",
    "UM": "Upper-\nMiddle\nIncome",
    "H": "High\nIncome",
    "": "Un-\nspecified",
}

country_col = "RecipientName"
if country_col not in df.columns:
    raise KeyError(f"Expected country column '{country_col}' not found in df.")
if "WB_Incomegroup" not in df.columns:
    raise KeyError("Expected column 'WB_Incomegroup' not found in df.")

df["WB_Incomegroup"] = df["WB_Incomegroup"].apply(_map_income_code).fillna("")

# 2) Filter to main analysis subset
df_income = df[
    (df["binary_label_biodiversity_impact"] == 1)
    & (df["No_Biodiv"] != 1)
    & (df["USD_Disbursement"] > 0)
    & (df["Year"].isin(years_of_interest))
    & (df["WB_Incomegroup"].isin(income_groups))
].copy()

avg_disb_per_income = (
    df_income.groupby("WB_Incomegroup")["USD_Disbursement"].sum() / len(years_of_interest)
).reindex(income_groups, fill_value=0)

# Compute with MERGED acts + normal impl/eco
act_rel_merged = compute_country_normalized_shares(
    df_income, subclass_list=act_merged_order
)
impl_rel = compute_country_normalized_shares(
    df_income, subclass_list=impl_order
)
eco_rel = compute_country_normalized_shares(
    df_income, subclass_list=eco_order
)


# ------------------------------------------------------------------------------
# SI: CORRELATION MATRICES (COLORED HEATMAPS)
# ------------------------------------------------------------------------------

# Custom green gradient
cmap = LinearSegmentedColormap.from_list('money', ['#F5F5F5', '#A9A9A9', '#2F4F4F'])

# Create reverse mapping from label text → column name
act_reverse_labels = {v: k for k, v in act_labels.items()}
impl_reverse_labels = {v: k for k, v in impl_labels.items()}
eco_reverse_labels = {v: k for k, v in eco_labels.items()}

def create_correlation_matrix(df, rows, cols, row_dim, col_dim):
    """
    Computes the average yearly disbursement and project counts between each pair of subcategories.

    Disbursements are split proportionally when multiple subcategories are tagged.
    Project counts are averaged over years and count only full co-occurrences.
    """
    df = df[(df['USD_Disbursement'] > 0)].copy()
    df = df[df[row_dim + col_dim].sum(axis=1) > 0]

    # Pre-compute how many 1's per row per dimension
    df['row_total'] = df[row_dim].sum(axis=1)
    df['col_total'] = df[col_dim].sum(axis=1)

    disb_matrix = pd.DataFrame(index=rows, columns=cols, dtype=float)
    count_matrix = pd.DataFrame(index=rows, columns=cols, dtype=float)

    # Iterate through each cell (row-category x col-category)
    for r in rows:
        for c in cols:
            # Filter projects that have both dimensions active
            both_active = df[(df[r] == 1) & (df[c] == 1)].copy()
            if both_active.empty:
                disb_matrix.loc[r, c] = 0
                count_matrix.loc[r, c] = 0
                continue

            # Apply proportional split
            both_active['disb_split'] = both_active['USD_Disbursement'] / (both_active['row_total'] * both_active['col_total'])

            # Sum by year
            yearly_totals = both_active.groupby('Year')['disb_split'].sum()
            avg_disb = yearly_totals.reindex(range(start_year_of_interest, end_year_of_interest+1), fill_value=0).mean()

            yearly_counts = both_active.groupby('Year').size()
            avg_count = yearly_counts.reindex(range(start_year_of_interest, end_year_of_interest+1), fill_value=0).mean()

            disb_matrix.loc[r, c] = avg_disb
            count_matrix.loc[r, c] = avg_count

    return disb_matrix.fillna(0), count_matrix.fillna(0).astype(int)


# Recompute subclass totals using intra-dimension-only splits
act_totals_corrected = compute_average_totals(df_filtered, act_cols, years_of_interest)
impl_totals_corrected = compute_average_totals(df_filtered, impl_cols, years_of_interest)
eco_totals_corrected = compute_average_totals(df_filtered, eco_cols, years_of_interest)

def add_totals_to_matrix(matrix, row_totals_override=None, col_totals_override=None):
    """
    Adds TOTAL row and column to the correlation matrix.
    If override dicts are passed, use them instead of the default sums.
    """
    matrix_with_totals = matrix.copy()

    # Column totals (sum of rows per column)
    col_totals = matrix.sum(axis=0) if col_totals_override is None else pd.Series(col_totals_override)
    matrix_with_totals.loc['TOTAL'] = col_totals

    # Row totals (sum of columns per row)
    row_totals = matrix.sum(axis=1) if row_totals_override is None else pd.Series(row_totals_override)
    matrix_with_totals['TOTAL'] = pd.concat([row_totals, pd.Series({'TOTAL': col_totals.sum()})])

    return matrix_with_totals


def move_total_to_end(df):
    df = df.copy()
    if 'TOTAL' in df.columns:
        cols = [col for col in df.columns if col != 'TOTAL'] + ['TOTAL']
        df = df[cols]
    if 'TOTAL' in df.index:
        rows = [row for row in df.index if row != 'TOTAL'] + ['TOTAL']
        df = df.loc[rows]
    return df

#print("\nCreating correlation matrices...")
act_impl_amount, act_impl_count = create_correlation_matrix(df_filtered, impl_cols, act_cols, impl_cols, act_cols)
act_eco_amount, act_eco_count = create_correlation_matrix(df_filtered, eco_cols, act_cols, eco_cols, act_cols)
eco_impl_amount, eco_impl_count = create_correlation_matrix(df_filtered, impl_cols, eco_cols, impl_cols, eco_cols)

act_impl_amount = add_totals_to_matrix(
    act_impl_amount,
    row_totals_override=impl_totals_corrected,
    col_totals_override=act_totals_corrected
)

act_eco_amount = add_totals_to_matrix(
    act_eco_amount,
    row_totals_override=eco_totals_corrected,
    col_totals_override=act_totals_corrected
)

eco_impl_amount = add_totals_to_matrix(
    eco_impl_amount,
    row_totals_override=impl_totals_corrected,
    col_totals_override=eco_totals_corrected
)

act_impl_count = add_totals_to_matrix(act_impl_count)
act_eco_count = add_totals_to_matrix(act_eco_count)
eco_impl_count = add_totals_to_matrix(eco_impl_count)

# Reorder so TOTAL is last row and column for better display
act_impl_amount = move_total_to_end(act_impl_amount)
act_impl_count = move_total_to_end(act_impl_count)

act_eco_amount = move_total_to_end(act_eco_amount)
act_eco_count = move_total_to_end(act_eco_count)

eco_impl_amount = move_total_to_end(eco_impl_amount)
eco_impl_count = move_total_to_end(eco_impl_count)


def exclude_total(df):
    return df.drop(index='TOTAL', errors='ignore').drop(columns='TOTAL', errors='ignore')

global_min = min(exclude_total(m).min().min() for m in [act_impl_amount, act_eco_amount, eco_impl_amount])
global_max = max(exclude_total(m).max().max() for m in [act_impl_amount, act_eco_amount, eco_impl_amount])


plt.figure(figsize=(30, 15))

def format_annotations(amount_df):
    annot = np.empty_like(amount_df.values, dtype=object)
    for i in range(amount_df.shape[0]):
        for j in range(amount_df.shape[1]):
            amount = amount_df.iloc[i, j]
            if amount >= 1000:
                display_amt = f"$\\bf{{{amount:,.0f}}}$"
            else:
                display_amt = f"$\\bf{{{amount:,.1f}}}$"
            annot[i, j] = display_amt
    return annot


fig, axs = plt.subplots(2, 2, figsize=(28, 16))
fig.delaxes(axs[1, 1])

# === 1. Action vs Implementation ===
ax = axs[0, 0]

# Prepare data and annotations
amount_df = act_impl_amount
count_df = act_impl_count
annot = format_annotations(amount_df)
xticks = np.arange(len(amount_df.columns)) + 0.5
yticks = np.arange(len(amount_df.index)) + 0.5


plot_amount_df = amount_df.copy()
mask = pd.DataFrame(False, index=plot_amount_df.index, columns=plot_amount_df.columns)
if 'TOTAL' in mask.index:
    mask.loc['TOTAL', :] = True
if 'TOTAL' in mask.columns:
    mask.loc[:, 'TOTAL'] = True

annot_masked = np.where(mask.values, '', annot)

sns.heatmap(
    plot_amount_df,
    annot=annot_masked,
    fmt='',
    cmap=cmap,
    ax=ax,
    linewidths=.5,
    vmin=global_min,
    vmax=global_max,
    cbar=False,
    annot_kws={"size": 18},
    xticklabels=amount_df.columns,
    yticklabels=amount_df.index,
    mask=mask
)

for i in range(amount_df.shape[0]):
    for j in range(amount_df.shape[1]):
        row_label = amount_df.index[i]
        col_label = amount_df.columns[j]
        if row_label == 'TOTAL' or col_label == 'TOTAL':
            text = annot[i, j]
            if pd.notna(amount_df.iloc[i, j]):
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=18, weight='bold')

ax.set_xticks(xticks)
ax.set_yticks(yticks)

ax.set_xlabel("Goal", fontsize=20, fontweight='bold', labelpad=20)
ax.set_ylabel("Instrument", fontsize=20,fontweight='bold', labelpad=20)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

xtick_labels = [
    act_labels.get(label.get_text(), label.get_text())
    if label.get_text() != 'TOTAL' else 'Total'
    for label in ax.get_xticklabels()
]
ytick_labels = [
    impl_labels.get(label.get_text(), label.get_text())
    if label.get_text() != 'TOTAL' else 'Total'
    for label in ax.get_yticklabels()
]

ax.set_xticklabels(xtick_labels, rotation=90, ha='center', fontsize=16)
ax.set_yticklabels(ytick_labels, rotation=0, fontsize=16)

# Define labels that should have white font
white_font_labels = {
    'Pollution\ncontrol',
    'Invasive\nspecies\nmanagement',
    'Sustainable\nagriculture',
    'Sustainable\nforest\nmanagement',
    'Sustainable\nfishery',
    'Sustainable\nwater\nmanagement',
    'Sustainable\nmgmt. of other\nnatural\nresources',
    'Sustainable\nresource\nmanagement'
}

for label in ax.get_xticklabels():
    text = label.get_text()
    if text == 'Total':
        continue  # Skip styling Total
    col = act_reverse_labels.get(text) or eco_reverse_labels.get(text)
    if col in act_order:
        label.set_backgroundcolor(act_colors[act_order.index(col)])
    elif col in eco_order:
        label.set_backgroundcolor(eco_colors[eco_order.index(col)])

    # Set font color to white if in defined list
    if text in white_font_labels:
        label.set_color('white')


for label in ax.get_yticklabels():
    text = label.get_text()
    col = impl_reverse_labels.get(text)
    if col in impl_order:
        label.set_backgroundcolor(impl_colors[impl_order.index(col)])

for label in ax.get_xticklabels():
    if label.get_text() == "Total":
        label.set_fontweight('bold')
        label.set_color('black')

for label in ax.get_yticklabels():
    if label.get_text() == "Total":
        label.set_fontweight('bold')
        label.set_color('black')


# === 2. Action vs Ecosystem ===
ax = axs[0, 1]

# Prepare data and annotations
amount_df = act_eco_amount
count_df = act_eco_count
annot = format_annotations(amount_df)
xticks = np.arange(len(amount_df.columns)) + 0.5
yticks = np.arange(len(amount_df.index)) + 0.5

# Make a fresh copy of the correct amount_df
plot_amount_df = amount_df.copy()

# Create mask for TOTAL row and column
mask = pd.DataFrame(False, index=plot_amount_df.index, columns=plot_amount_df.columns)
if 'TOTAL' in mask.index:
    mask.loc['TOTAL', :] = True
if 'TOTAL' in mask.columns:
    mask.loc[:, 'TOTAL'] = True

annot_masked = np.where(mask.values, '', annot)

sns.heatmap(
    plot_amount_df,
    annot=annot_masked,
    fmt='',
    cmap=cmap,
    ax=ax,
    linewidths=.5,
    vmin=global_min,
    vmax=global_max,
    cbar=False,
    annot_kws={"size": 18},
    xticklabels=amount_df.columns,
    yticklabels=amount_df.index,
    mask=mask
)

for i in range(amount_df.shape[0]):
    for j in range(amount_df.shape[1]):
        row_label = amount_df.index[i]
        col_label = amount_df.columns[j]
        if row_label == 'TOTAL' or col_label == 'TOTAL':
            text = annot[i, j]
            if pd.notna(amount_df.iloc[i, j]):
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=18, weight='bold')

ax.set_xticks(xticks)
ax.set_yticks(yticks)

ax.set_xlabel("Goal", fontsize=20, fontweight='bold',labelpad=20)
ax.set_ylabel("Ecosystem", fontsize=20, fontweight='bold',labelpad=20)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

xtick_labels = [
    act_labels.get(label.get_text(), label.get_text())
    if label.get_text() != 'TOTAL' else 'Total'
    for label in ax.get_xticklabels()
]
ytick_labels = [
    eco_labels.get(label.get_text(), label.get_text())
    if label.get_text() != 'TOTAL' else 'Total'
    for label in ax.get_yticklabels()
]

ax.set_xticklabels(xtick_labels, rotation=90, ha='center', fontsize=16)
ax.set_yticklabels(ytick_labels, rotation=0, fontsize=16)

# Define labels that should have white font
white_font_labels = {
    'Pollution\ncontrol',
    'Invasive\nspecies\nmanagement',
    'Sustainable\nagriculture',
    'Sustainable\nforest\nmanagement',
    'Sustainable\nfishery',
    'Sustainable\nwater\nmanagement',
    'Sustainable\nmgmt. of other\nnatural\nresources',
    'Sustainable\nresource\nmanagement'
}

for label in ax.get_xticklabels():
    text = label.get_text()
    if text == 'Total':
        continue
    col = act_reverse_labels.get(text) or eco_reverse_labels.get(text)
    if col in act_order:
        label.set_backgroundcolor(act_colors[act_order.index(col)])
    elif col in eco_order:
        label.set_backgroundcolor(eco_colors[eco_order.index(col)])

    if text in white_font_labels:
        label.set_color('white')

for label in ax.get_yticklabels():
    text = label.get_text()
    col = eco_reverse_labels.get(text)
    if col in eco_order:
        label.set_backgroundcolor(eco_colors[eco_order.index(col)])

for label in ax.get_xticklabels():
    if label.get_text() == "Total":
        label.set_fontweight('bold')
        label.set_color('black')

for label in ax.get_yticklabels():
    if label.get_text() == "Total":
        label.set_fontweight('bold')
        label.set_color('black')


# === 3. Ecosystem vs Implementation ===
ax = axs[1, 0]

# Prepare data and annotations
amount_df = eco_impl_amount
count_df = eco_impl_count
annot = format_annotations(amount_df)
xticks = np.arange(len(amount_df.columns)) + 0.5
yticks = np.arange(len(amount_df.index)) + 0.5

# Make a fresh copy of the correct amount_df
plot_amount_df = amount_df.copy()

# Create mask for TOTAL row and column
mask = pd.DataFrame(False, index=plot_amount_df.index, columns=plot_amount_df.columns)
if 'TOTAL' in mask.index:
    mask.loc['TOTAL', :] = True
if 'TOTAL' in mask.columns:
    mask.loc[:, 'TOTAL'] = True

annot_masked = np.where(mask.values, '', annot)

sns.heatmap(
    plot_amount_df,
    annot=annot_masked,
    fmt='',
    cmap=cmap,
    ax=ax,
    linewidths=.5,
    vmin=global_min,
    vmax=global_max,
    cbar=False,
    annot_kws={"size": 18},
    xticklabels=amount_df.columns,
    yticklabels=amount_df.index,
    mask=mask
)

for i in range(amount_df.shape[0]):
    for j in range(amount_df.shape[1]):
        row_label = amount_df.index[i]
        col_label = amount_df.columns[j]
        if row_label == 'TOTAL' or col_label == 'TOTAL':
            text = annot[i, j]
            if pd.notna(amount_df.iloc[i, j]):
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=18, weight='bold')

ax.set_xticks(xticks)
ax.set_yticks(yticks)

ax.set_xlabel("Ecosystem", fontsize=20, fontweight='bold', labelpad=20)
ax.set_ylabel("Instrument", fontsize=20,fontweight='bold', labelpad=20)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

xtick_labels = [
    eco_labels.get(label.get_text(), label.get_text())
    if label.get_text() != 'TOTAL' else 'Total'
    for label in ax.get_xticklabels()
]
ytick_labels = [
    impl_labels.get(label.get_text(), label.get_text())
    if label.get_text() != 'TOTAL' else 'Total'
    for label in ax.get_yticklabels()
]

ax.set_xticklabels(xtick_labels, rotation=90, ha='center', fontsize=16)
ax.set_yticklabels(ytick_labels, rotation=0, fontsize=16)

for label in ax.get_xticklabels():
    text = label.get_text()
    if text == 'Total':
        continue  # Skip styling Total
    col = eco_reverse_labels.get(text) or impl_reverse_labels.get(text)
    if col in eco_order:
        label.set_backgroundcolor(eco_colors[eco_order.index(col)])

for label in ax.get_yticklabels():
    text = label.get_text()
    col = impl_reverse_labels.get(text)
    if col in impl_order:
        label.set_backgroundcolor(impl_colors[impl_order.index(col)])

for label in ax.get_xticklabels():
    if label.get_text() == "Total":
        label.set_fontweight('bold')
        label.set_color('black')

for label in ax.get_yticklabels():
    if label.get_text() == "Total":
        label.set_fontweight('bold')
        label.set_color('black')

# === Horizontal legend placement ===
norm = plt.Normalize(vmin=global_min, vmax=global_max)
smn = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
smn.set_array([])

cax = fig.add_axes([0.78, 0.08, 0.18, 0.02])
cbar = fig.colorbar(smn, cax=cax, orientation='horizontal')
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.xaxis.set_tick_params(top=False, bottom=True)
cbar.ax.set_xlabel(f"Annual USD bn Disbursement ({start_year_of_interest}-{end_year_of_interest} average)", fontsize=18, labelpad=6)
cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# === Layout Adjustments ===
fig.tight_layout()
fig.subplots_adjust(hspace=0.7, wspace=0.3)
output_path= "outputs/SI_combined_correlation_matrices_2D.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig('outputs/SI_combined_correlation_matrices_2D.pdf', dpi=300, bbox_inches='tight')
print(f"Saved correlation matrix figure to: {output_path} and .pdf")

#---------------------------------------------
#CORRELATION MATRICES FOR MERGED GOAL
#---------------------------------------------

# === STEP 1: Create Act_SustResMgmt Column ===
df['Act_SustResMgmt'] = df[['Act_Agri','Act_ForestMgmt','Act_Fish','Act_WaterMgmt','Act_OthMgmt']].max(axis=1)
df['Act_Protect_Resto'] = df[['Act_Protect', 'Act_Resto']].max(axis=1)

act_merged_reverse_labels = {v: k for k, v in act_merged_labels.items()}

# === STEP 3: Recompute correlation matrices using merged column list ===
act_totals_corrected = compute_average_totals(df, act_merged_cols, years_of_interest)

# ACTION vs IMPLEMENTATION
act_impl_amount, act_impl_count = create_correlation_matrix(df, impl_cols, act_merged_cols, impl_cols, act_merged_cols)

# ACTION vs ECOSYSTEM
act_eco_amount, act_eco_count = create_correlation_matrix(df, eco_cols, act_merged_cols, eco_cols, act_merged_cols)

# ECOSYSTEM vs IMPLEMENTATION
eco_impl_amount, eco_impl_count = create_correlation_matrix(df, impl_cols, eco_cols, impl_cols, eco_cols)

# Add totals
act_impl_amount = add_totals_to_matrix(
    act_impl_amount,
    row_totals_override=impl_totals_corrected,
    col_totals_override=act_totals_corrected
)

act_eco_amount = add_totals_to_matrix(
    act_eco_amount,
    row_totals_override=eco_totals_corrected,
    col_totals_override=act_totals_corrected
)

eco_impl_amount = add_totals_to_matrix(
    eco_impl_amount,
    row_totals_override=impl_totals_corrected,
    col_totals_override=eco_totals_corrected
)

# Add totals to count matrices
act_impl_count = add_totals_to_matrix(act_impl_count)
act_eco_count = add_totals_to_matrix(act_eco_count)
eco_impl_count = add_totals_to_matrix(eco_impl_count)

# Move TOTAL row/col to end
act_impl_amount = move_total_to_end(act_impl_amount)
act_impl_count = move_total_to_end(act_impl_count)

act_eco_amount = move_total_to_end(act_eco_amount)
act_eco_count = move_total_to_end(act_eco_count)

eco_impl_amount = move_total_to_end(eco_impl_amount)
eco_impl_count = move_total_to_end(eco_impl_count)

# Update global min/max for consistent color scaling
global_min = min(exclude_total(m).min().min() for m in [act_impl_amount, act_eco_amount, eco_impl_amount])
global_max = max(exclude_total(m).max().max() for m in [act_impl_amount, act_eco_amount, eco_impl_amount])

# === STEP 4: Plot (reuse full plotting code) ===
# === Final Plotting Code ===

fig, axs = plt.subplots(2, 2, figsize=(28, 16))
fig.delaxes(axs[1, 1])  # Remove unused subplot

# Helper function to clean totals
def exclude_total(df):
    return df.drop(index='TOTAL', errors='ignore').drop(columns='TOTAL', errors='ignore')

# Compute global color range
global_min = min(exclude_total(m).min().min() for m in [act_impl_amount, act_eco_amount, eco_impl_amount])
global_max = max(exclude_total(m).max().max() for m in [act_impl_amount, act_eco_amount, eco_impl_amount])

# Annotation formatter
def format_annotations(amount_df):
    annot = np.empty_like(amount_df.values, dtype=object)
    for i in range(amount_df.shape[0]):
        for j in range(amount_df.shape[1]):
            amount = amount_df.iloc[i, j]
            if amount >= 1000:
                display_amt = f"$\\bf{{{amount:,.0f}}}$"
            else:
                display_amt = f"$\\bf{{{amount:,.1f}}}$"
            annot[i, j] = display_amt
    return annot

# Custom green gradient colormap

cmap = LinearSegmentedColormap.from_list('money', ['#F5F5F5', '#A9A9A9', '#2F4F4F'])

# === Plot Panels ===
matrix_configs = [
    (act_impl_amount, act_impl_count, impl_labels, act_merged_labels, "Instrument", "Goal"),
    (act_eco_amount, act_eco_count, eco_labels, act_merged_labels, "Ecosystem", "Goal"),
    (eco_impl_amount, eco_impl_count, impl_labels, eco_labels, "Instrument", "Ecosystem")
]

for idx, (amount_df, count_df, y_labels_map, x_labels_map, y_label, x_label) in enumerate(matrix_configs):
    ax = axs[idx // 2, idx % 2]

    annot = format_annotations(amount_df)
    xticks = np.arange(len(amount_df.columns)) + 0.5
    yticks = np.arange(len(amount_df.index)) + 0.5

    plot_amount_df = amount_df.copy()
    mask = pd.DataFrame(False, index=plot_amount_df.index, columns=plot_amount_df.columns)
    if 'TOTAL' in mask.index:
        mask.loc['TOTAL', :] = True
    if 'TOTAL' in mask.columns:
        mask.loc[:, 'TOTAL'] = True
    annot_masked = np.where(mask.values, '', annot)

    sns.heatmap(
        plot_amount_df,
        annot=annot_masked,
        fmt='',
        cmap=cmap,
        ax=ax,
        linewidths=.5,
        vmin=global_min,
        vmax=global_max,
        cbar=False,
        annot_kws={"size": 18},
        xticklabels=amount_df.columns,
        yticklabels=amount_df.index,
        mask=mask
    )

    # TOTAL annotations
    for i in range(amount_df.shape[0]):
        for j in range(amount_df.shape[1]):
            if amount_df.index[i] == 'TOTAL' or amount_df.columns[j] == 'TOTAL':
                if pd.notna(amount_df.iloc[i, j]):
                    ax.text(j + 0.5, i + 0.5, annot[i, j], ha='center', va='center', fontsize=18, weight='bold')

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(x_label, fontsize=20, fontweight='bold',labelpad=20)
    ax.set_ylabel(y_label, fontsize=20, fontweight='bold',labelpad=20)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Update tick labels
    ax.set_xticklabels(
        [x_labels_map.get(t.get_text(), t.get_text()) if t.get_text() != 'TOTAL' else 'Total'
         for t in ax.get_xticklabels()],
        rotation=90, ha='center', fontsize=16
    )
    ax.set_yticklabels(
        [y_labels_map.get(t.get_text(), t.get_text()) if t.get_text() != 'TOTAL' else 'Total'
         for t in ax.get_yticklabels()],
        rotation=0, fontsize=16
    )

    # === Apply background colors to tick labels ===
    for label in ax.get_xticklabels():
        text = label.get_text()
        if text == 'Total':
            continue
        col = None
        if x_labels_map is act_merged_labels:
            col = act_merged_reverse_labels.get(text)
            if col in act_merged_order:
                label.set_backgroundcolor(act_merged_colors[act_merged_order.index(col)])
        elif x_labels_map is eco_labels:
            col = eco_reverse_labels.get(text)
            if col in eco_order:
                label.set_backgroundcolor(eco_colors[eco_order.index(col)])

        if text in white_font_labels:
            label.set_color('white')

    for label in ax.get_yticklabels():
        text = label.get_text()
        col = None
        if y_labels_map is impl_labels:
            col = impl_reverse_labels.get(text)
            if col in impl_order:
                label.set_backgroundcolor(impl_colors[impl_order.index(col)])
        elif y_labels_map is eco_labels:
            col = eco_reverse_labels.get(text)
            if col in eco_order:
                label.set_backgroundcolor(eco_colors[eco_order.index(col)])

    for label in ax.get_xticklabels():
        if label.get_text() == "Total":
            label.set_fontweight('bold')
            label.set_color('black')

    for label in ax.get_yticklabels():
        if label.get_text() == "Total":
            label.set_fontweight('bold')
            label.set_color('black')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if label.get_text() == "Total":
            label.set_fontweight('bold')
            label.set_color('black')

# === Horizontal colorbar ===
norm = plt.Normalize(vmin=global_min, vmax=global_max)
smn = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
smn.set_array([])

cax = fig.add_axes([0.78, 0.08, 0.18, 0.02])
cbar = fig.colorbar(smn, cax=cax, orientation='horizontal')
cbar.ax.set_xlabel(f"Annual USD bn Disbursement ({start_year_of_interest}–{end_year_of_interest} average)", fontsize=18, labelpad=6)
cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


# ------------------------------------------------------------------------------
# TABLE: IMPLEMENTATION PERCENTAGE SHARES PER ACT–ECO COMBINATION (years of interest average)
# ------------------------------------------------------------------------------

# Ensure merged ACT column is present
if 'Act_SustResMgmt' not in df.columns:
    df['Act_SustResMgmt'] = df[['Act_Agri','Act_ForestMgmt','Act_Fish','Act_WaterMgmt','Act_OthMgmt']].max(axis=1)

if 'Act_Protect_Resto' not in df.columns:
        df['Act_Protect_Resto'] = df[['Act_Protect', 'Act_Resto']].max(axis=1)



# Only use rows from years of interest with valid disbursements
df_avg = df[(df['Year'].isin(years_of_interest)) & (df['USD_Disbursement'] > 0)].copy()

# Initialize list to store output rows
impl_share_rows = []

# Loop through Act × Eco combinations
for act_col in act_merged_cols:
    for eco_col in eco_cols:
        subset = df_avg[(df_avg[act_col] == 1) & (df_avg[eco_col] == 1)]

        if subset.empty:
            impl_share_rows.append({
                'Action': act_col,
                'Ecosystem': eco_col,
                'Impl_Regul': 0.0,
                'Impl_Know': 0.0,
                'Impl_Infra': 0.0,
                'Impl_Undef': 0.0
            })
            continue

        # Initialize disbursement accumulators
        impl_disb = {col: 0.0 for col in impl_cols}
        total_disb = 0.0

        for _, row in subset.iterrows():
            active_impls = [col for col in impl_cols if row[col] == 1]
            if not active_impls:
                continue

            disb_share = row['USD_Disbursement'] / len(active_impls)
            for col in active_impls:
                impl_disb[col] += disb_share
            total_disb += row['USD_Disbursement']

        # Compute percentage shares
        impl_pct = {
            col: (impl_disb[col] / total_disb * 100) if total_disb > 0 else 0.0
            for col in impl_cols
        }

        impl_share_rows.append({
            'Action': act_col,
            'Ecosystem': eco_col,
            **impl_pct
        })

# Convert to DataFrame
impl_share_df = pd.DataFrame(impl_share_rows)

# Optional: Apply readable labels
impl_share_df['Action'] = impl_share_df['Action'].map(act_merged_labels_nobreak)
impl_share_df['Ecosystem'] = impl_share_df['Ecosystem'].map(eco_labels_nobreak)

# Round values to 2 decimals for export
for col in impl_cols:
    impl_share_df[col] = impl_share_df[col].round(4)

# Export to Excel
output_path = "outputs/act_eco_impl_disbursement_percentage_shares.xlsx"
impl_share_df.to_excel(output_path, index=False)

print(f"Saved disbursement-weighted implementation shares to {output_path}")


# ------------------------------------------------------------------------------
# CONFIGURATION FOR 3D ACT–IMPL–ECO AVERAGES
# ------------------------------------------------------------------------------

def compute_and_export_3d_act_impl_eco_avgs(
    df,
    years_of_interest,
    act_merged_cols,
    impl_cols,
    eco_cols,
    output_path="outputs/act_impl_eco_3D_annual_avg_disbursements.xlsx"
):

    # Ensure merged ACT columns exist (same definitions used above)
    if 'Act_SustResMgmt' not in df.columns:
        df['Act_SustResMgmt'] = df[['Act_Agri','Act_ForestMgmt','Act_Fish','Act_WaterMgmt','Act_OthMgmt']].max(axis=1)

    if 'Act_Protect_Resto' not in df.columns:
        df['Act_Protect_Resto'] = df[['Act_Protect', 'Act_Resto']].max(axis=1)

    # Filter to years of interest and positive disbursements
    df_3d = df[(df['Year'].isin(years_of_interest)) & (df['USD_Disbursement'] > 0)].copy()

    # Keep only rows with at least one class active in each dimension
    df_3d = df_3d[
        (df_3d[act_merged_cols].sum(axis=1) > 0) &
        (df_3d[impl_cols].sum(axis=1) > 0) &
        (df_3d[eco_cols].sum(axis=1) > 0)
    ].copy()

    # (Act, Impl, Eco) -> {year -> disbursement_sum}
    triple_year_disb = defaultdict(lambda: defaultdict(float))

    # Proportional split across ALL active classes in EACH dimension
    for _, row in df_3d.iterrows():
        active_act_cols = [col for col in act_merged_cols if row[col] == 1]
        active_impl_cols = [col for col in impl_cols if row[col] == 1]
        active_eco_cols = [col for col in eco_cols if row[col] == 1]

        if not active_act_cols or not active_impl_cols or not active_eco_cols:
            continue

        n_act = len(active_act_cols)
        n_impl = len(active_impl_cols)
        n_eco = len(active_eco_cols)

        denom = n_act * n_impl * n_eco
        if denom == 0:
            continue

        disb_share = row['USD_Disbursement'] / denom
        year = int(row['Year'])

        for act_col in active_act_cols:
            for impl_col in active_impl_cols:
                for eco_col in active_eco_cols:
                    triple_year_disb[(act_col, impl_col, eco_col)][year] += disb_share

    # Build output: annual average of years of interest for each triple
    output_rows = []

    for (act_col, impl_col, eco_col), year_dict in triple_year_disb.items():
        total_years_of_interest = sum(year_dict.get(y, 0.0) for y in years_of_interest)
        annual_avg = total_years_of_interest / len(years_of_interest)

        output_rows.append({
            'Act_Class': act_col,
            'Impl_Class': impl_col,
            'Eco_Class': eco_col,
            'Annual_Avg': annual_avg
        })

    triple_df = pd.DataFrame(output_rows)

    # --- Optional: map to readable labels if dicts exist in environment ---

    def safe_map(series, mapping_name, fallback_mapping_name=None):
        mapping = globals().get(mapping_name) or globals().get(fallback_mapping_name, {})
        return series.map(lambda x: mapping.get(x, x))

    # Act labels
    if 'act_merged_labels_nobreak' in globals() or 'act_merged_labels' in globals():
        triple_df['Act_Label'] = safe_map(
            triple_df['Act_Class'],
            'act_merged_labels_nobreak',
            'act_merged_labels'
        )
    else:
        triple_df['Act_Label'] = triple_df['Act_Class']

    # Impl labels
    if 'impl_labels_nobreak' in globals() or 'impl_labels' in globals():
        triple_df['Impl_Label'] = safe_map(
            triple_df['Impl_Class'],
            'impl_labels_nobreak',
            'impl_labels'
        )
    else:
        triple_df['Impl_Label'] = triple_df['Impl_Class']

    # Eco labels
    if 'eco_labels_nobreak' in globals() or 'eco_labels' in globals():
        triple_df['Eco_Label'] = safe_map(
            triple_df['Eco_Class'],
            'eco_labels_nobreak',
            'eco_labels'
        )
    else:
        triple_df['Eco_Label'] = triple_df['Eco_Class']

    # Reorder columns
    label_cols = ['Act_Label', 'Impl_Label', 'Eco_Label']
    code_cols = ['Act_Class', 'Impl_Class', 'Eco_Class']
    metric_cols = ['Annual_Avg']

    triple_df = triple_df[label_cols + code_cols + metric_cols]
    triple_df = triple_df.sort_values(by=['Act_Label', 'Impl_Label', 'Eco_Label']).reset_index(drop=True)

    # Export to Excel
    triple_df.to_excel(output_path, index=False)
    print(f"SI Table: Saved annual averages for merged_act × impl × eco intersections to {output_path}")

    return triple_df


triple_df = compute_and_export_3d_act_impl_eco_avgs(
    df=df,
    years_of_interest=years_of_interest,
    act_merged_cols=act_merged_cols,
    impl_cols=impl_cols,
    eco_cols=eco_cols
)

# ---------------------------------------
# SI: SCATTERPLOTS BIODIVERSITY INDEX (ISO3-based)
# ---------------------------------------

# === Global style tweaks ===
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 15

# Ensure output folder exists
os.makedirs(
    "./outputs", exist_ok=True)

# === Get ISO3 Codes ===
def get_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

# Add ISO3 column to df
df['ISO3'] = df['RecipientName'].apply(get_iso3)


# === Load epi2024results.csv ===
epi = pd.read_csv(   "inputs/epi2024results.csv", sep=';')
epi['iso'] = epi['iso'].str.upper().str.strip()

# === Filter to matched countries with biodiversity impact ===
df_matched = df[
    (df['ISO3'].notna()) &
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1)
].copy()
df_matched['ISO3'] = df_matched['ISO3'].str.upper()

# === Aggregate USD_Disbursement per ISO3 country (2000–2023) ===
agg_disb = df_matched.groupby("ISO3")["USD_Disbursement"].sum().reset_index()
agg_disb.columns = ["ISO3", "USD_Disbursement"]

# === Merge with epi data ===
merged = agg_disb.merge(epi, left_on="ISO3", right_on="iso", how="left")

# === BII data ===
bii = pd.read_excel("inputs/BII_National history museum.xlsx")
bii['iso'] = bii['iso'].str.upper().str.strip()
bii = bii[['iso', 'value']].rename(columns={'iso': 'ISO3', 'value': 'Biodiversity_Intactness'})
merged_bii = agg_disb.merge(bii, on="ISO3", how="left")
merged_bii = merged_bii.merge(epi[['iso', 'country']], left_on="ISO3", right_on="iso", how="left")

# === Species Richness Data ===
species = pd.read_excel("inputs/Map of Life_Species Richness.xlsx")
species['RecipientName'] = species['RecipientName'].str.strip()
species['ISO3'] = species['RecipientName'].apply(get_iso3)

species = species[['ISO3', 'Species present']].dropna()
species['ISO3'] = species['ISO3'].str.upper()
merged_species = agg_disb.merge(species, on='ISO3', how='left')
merged_species = merged_species.merge(epi[['iso', 'country']], left_on='ISO3', right_on='iso', how='left')

# === Brancalion (Restoration) Data ===
branc = pd.read_excel("inputs/Bracilion_2019.xlsx")
branc['Country'] = branc['Country'].astype(str).str.strip()
branc['ISO3'] = branc['Country'].apply(get_iso3)

branc = branc[['ISO3', 'Restoration_hotspot_area_Mha', 'Restoration_opportunity_score_mean']].copy()
branc['ISO3'] = branc['ISO3'].str.upper()

merged_rest_area = agg_disb.merge(branc[['ISO3', 'Restoration_hotspot_area_Mha']], on='ISO3', how='left')
merged_rest_area = merged_rest_area.merge(epi[['iso', 'country']], left_on='ISO3', right_on='iso', how='left')

merged_rest_score = agg_disb.merge(branc[['ISO3', 'Restoration_opportunity_score_mean']], on='ISO3', how='left')
merged_rest_score = merged_rest_score.merge(epi[['iso', 'country']], left_on='ISO3', right_on='iso', how='left')

# === NBI data ===
nbi = pd.read_excel("inputs/NBI_score.xlsx")
nbi['Country'] = nbi['Country'].astype(str).str.strip()
nbi['ISO3'] = nbi['Country'].apply(get_iso3)

nbi = nbi[['ISO3', 'NBI']].copy()
nbi['ISO3'] = nbi['ISO3'].str.upper()
merged_nbi = agg_disb.merge(nbi, on='ISO3', how='left')
merged_nbi = merged_nbi.merge(epi[['iso', 'country']], left_on='ISO3', right_on='iso', how='left')

# === STAR-T data ===
star = pd.read_excel("inputs/Star_T_Score_extended_ZSR_1408.xlsx")
star['RecipientName'] = star['RecipientName'].astype(str).str.strip()
star['ISO3'] = star['RecipientName'].apply(get_iso3)
score_col = "% Global Threat Abatement Score"
star[score_col] = (
    star[score_col].astype(str)
    .str.replace('%', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
star[score_col] = pd.to_numeric(star[score_col], errors='coerce')
mx = star[score_col].max(skipna=True)
if pd.notna(mx) and mx <= 1:
    star[score_col] = star[score_col] * 100
star = star[['ISO3', score_col]].dropna(subset=['ISO3', score_col])
star['ISO3'] = star['ISO3'].str.upper()
merged_star = agg_disb.merge(star, on='ISO3', how='left')
merged_star = merged_star.merge(epi[['iso', 'country']], left_on='ISO3', right_on='iso', how='left')

# === STAR-R data ===
star_r = pd.read_excel("inputs/STAR_R_by_country_normal_stats.xlsx")
star_r['name'] = star_r['name'].astype(str).str.strip()
star_r['ISO3'] = star_r['name'].apply(get_iso3)
star_r['STAR_R_mean'] = pd.to_numeric(star_r['STAR_R_mean'], errors='coerce')
star_r = star_r[['ISO3', 'STAR_R_mean']].dropna(subset=['ISO3', 'STAR_R_mean'])
star_r['ISO3'] = star_r['ISO3'].str.upper()
merged_star_r = agg_disb.merge(star_r, on='ISO3', how='left')
merged_star_r = merged_star_r.merge(epi[['iso', 'country']], left_on='ISO3', right_on='iso', how='left')

# === Create 3x4 subplot grid ===
fig, axs = plt.subplots(4, 3, figsize=(22, 30))
#plt.subplots_adjust(wspace=0.2, hspace=0.2)

# Utility for plotting
def make_scatter(ax, x, y, xlabel, ylabel, title, logx=False):
    ax.scatter(x, y, s=30, color='grey')
    ax.set_xlabel(xlabel, fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=18, fontweight='bold', loc='left')
    if logx:
        ax.set_xscale('log')
    ax.grid(True)

# === Plot assignments ===
make_scatter(axs[0,0], merged_species["Species present"], merged_species["USD_Disbursement"],
             "Number of unique species present (Map of Life)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "a     Biodiversity Richness\n")

make_scatter(axs[0,1], merged["SHI.new"], merged["USD_Disbursement"],
             "Habitat Loss Index (SHI, EPI Yale)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "b     Biodiversity Loss\n")

make_scatter(axs[0,2], merged_bii["Biodiversity_Intactness"], merged_bii["USD_Disbursement"],
             "Biodiversity Intactness Index (BII, National History Museum)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "c     Biodiversity Intactness\n")

make_scatter(axs[1,0], merged["FLI.new"], merged["USD_Disbursement"],
             "Forest Landscape Integrity (FLI, EPI Yale)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "d     Forest Landscape Integrity\n")

make_scatter(axs[1,1], merged["BDH.new"], merged["USD_Disbursement"],
             "Biodiversity Action for Protection (BDH, EPI Yale)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "e     Biodiversity Action\n")

make_scatter(axs[1,2], merged_rest_area["Restoration_hotspot_area_Mha"], merged_rest_area["USD_Disbursement"],
             "Restoration hotspot area [Mha] (Brancalion 2019)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "f     Restoration Hotspot Area\n")

make_scatter(axs[2,0], merged_rest_score["Restoration_opportunity_score_mean"], merged_rest_score["USD_Disbursement"],
             "Restoration opportunity score (Brancalion 2019)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "g     Restoration Opportunity\n")

make_scatter(axs[2,1], merged_nbi["NBI"], merged_nbi["USD_Disbursement"],
             "National Biodiversity Index (NBI)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "h     Biodiversity State\n")

make_scatter(axs[2,2], merged_star[score_col], merged_star["USD_Disbursement"],
             "Global Threat Abatement Score [%] (STAR-T)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "i     Threat Abatement\n")

make_scatter(axs[3,0], merged_star_r["STAR_R_mean"], merged_star_r["USD_Disbursement"],
             "Restoration potential (log STAR-R, mean)",
             "Total USD Disbursement (USD bn, 2000–2023)",
             "j     Restoration Potential\n",
             logx=True)

# Hide the two remaining empty subplots
axs[3,1].axis('off')
axs[3,2].axis('off')

# === Final layout & save ===
plt.tight_layout(pad=3.0, w_pad=3.5, h_pad=4.0)
output_path = "outputs/SI_Biodiversity_Index_Analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig('outputs/SI_Biodiversity_Index_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"SI Figure: Biodiversity Index analysis saved to {output_path} and .pdf")

# ---------------------------------------
# TABLE: REGRESSION ANALYSIS: Biodiversity Disbursement ~ Indicator Score
# ---------------------------------------

# Helper function for regressions
def run_regression(df, x_col, y_col="USD_Disbursement", label=""):
    """Run OLS regression of y_col on x_col and return summary stats."""
    if x_col not in df.columns:
        return None

    sub = df[[x_col, y_col]].dropna()
    if sub.empty or sub[x_col].nunique() < 2:
        return None

    X = sm.add_constant(sub[x_col])
    y = sub[y_col]
    model = sm.OLS(y, X).fit()
    return {
        "Indicator": label,
        "N": len(sub),
        "Slope": model.params.get(x_col, np.nan),
        "Intercept": model.params.get("const", np.nan),
        "R_squared": model.rsquared,
        "p_value": model.pvalues.get(x_col, np.nan)
    }

# List of regressions to run (dataframe, x-column, label)
regression_specs = [
    (merged_species, "Species present", "(a) Biodiversity Richness"),
    (merged, "SHI.new", "(b) Biodiversity Loss (SHI)"),
    (merged_bii, "Biodiversity_Intactness", "(c) Biodiversity Intactness (BII)"),
    (merged, "FLI.new", "(d) Forest Landscape Integrity (FLI)"),
    (merged, "BDH.new", "(e) Biodiversity Action (BDH)"),
    (merged_rest_area, "Restoration_hotspot_area_Mha", "(f) Restoration Hotspot Area"),
    (merged_rest_score, "Restoration_opportunity_score_mean", "(g) Restoration Opportunity Score"),
    (merged_nbi, "NBI", "(h) Biodiversity State (NBI)"),
    (merged_star, score_col, "(i) Threat Abatement (STAR-T)"),
    (merged_star_r, "STAR_R_mean", "(j) Restoration Potential (STAR-R)"),
]

# Run all regressions and collect results
regression_results = []
for df_, xcol, label in regression_specs:
    res = run_regression(df_, xcol, label=label)
    if res:
        regression_results.append(res)

# Convert to DataFrame
reg_df = pd.DataFrame(regression_results)
reg_df = reg_df[["Indicator", "N", "Slope", "Intercept", "R_squared", "p_value"]]

# Export results
out_reg_xlsx = "outputs/SI_regression_results_per_index.xlsx"
reg_df.to_excel(out_reg_xlsx, index=False)
print(f"SI Table: Regression summary written to {out_reg_xlsx}")

# ----------------------------------
# CONFIGURATION STAR-R
# ----------------------------------

PE = [withStroke(linewidth=3, foreground="white")]


def get_iso3(name):
    try:
        return pycountry.countries.lookup(str(name)).alpha_3
    except Exception:
        return None

color_by_key = dict(zip(act_merged_order, act_merged_colors))
label_by_key = act_merged_labels_nobreak

# Country labels (nice names)
epi = pd.read_csv(
    "inputs/epi2024results.csv", sep=";")
epi["iso"] = epi["iso"].str.upper().str.strip()


# Base data prep (ACT merge & proportional split)

# df must exist with: RecipientName, Year, USD_Disbursement, ACT columns used in act_merged_cols
df_work = df.copy()

# 1) Ensure merged ACT flags exist
if "Act_SustResMgmt" not in df_work.columns:
    df_work["Act_SustResMgmt"] = df_work[["Act_Agri","Act_ForestMgmt","Act_Fish","Act_WaterMgmt","Act_OthMgmt"]].max(axis=1)
if "Act_Protect_Resto" not in df_work.columns:
    df_work["Act_Protect_Resto"] = df_work[["Act_Protect","Act_Resto"]].max(axis=1)

# 2) Filter to years and clean disbursements
df_merged_act = df_work[df_work["Year"].isin(years_of_interest)].copy()
df_merged_act["USD_Disbursement"] = pd.to_numeric(df_merged_act["USD_Disbursement"], errors="coerce").fillna(0)
df_merged_act.loc[df_merged_act["USD_Disbursement"] < 0, "USD_Disbursement"] = 0
df_merged_act["USD_Disbursement"] *= 1000  # From bio → mio

# Ensure ACT columns are integers
df_merged_act[act_merged_cols] = df_merged_act[act_merged_cols].fillna(0).astype(int)

# Keep rows with at least one ACT label
df_merged_act["act_count"] = df_merged_act[act_merged_cols].sum(axis=1)
df_merged_act = df_merged_act[df_merged_act["act_count"] > 0].copy()

# Proportional split across merged ACTs (your approach)
for col in act_merged_cols:
    df_merged_act[f"{col}_share"] = df_merged_act[col] * df_merged_act["USD_Disbursement"] / df_merged_act["act_count"]

# ISO3 conversion (country mapping)
if "ISO3" not in df_merged_act.columns:
    df_merged_act["ISO3"] = df_merged_act["RecipientName"].apply(get_iso3)
df_merged_act = df_merged_act[df_merged_act["ISO3"].notna()].copy()
df_merged_act["ISO3"] = df_merged_act["ISO3"].str.upper()

# 3) Aggregate to compute average annual disbursement per ISO3 (years of interest) using split
# Sum the shares per (ISO3, Year) across all merged ACTs, then mean across years
share_cols = [f"{c}_share" for c in act_merged_cols]

annual_by_iso3_year = (
    df_merged_act
    .groupby(["ISO3", "Year"], as_index=False)[share_cols]
    .sum()
)
annual_by_iso3_year[f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}"] = annual_by_iso3_year[share_cols].sum(axis=1)

avg_annual = (
    annual_by_iso3_year
    .groupby("ISO3", as_index=False)[f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}"]
    .mean()
)


# STAR-R metrics

star_r = pd.read_excel("inputs/STAR_R_by_country_normal_stats.xlsx")  # expects: 'name', 'STAR_R_centiSTAR', 'STAR_R_mean'
star_r["name"] = star_r["name"].astype(str).str.strip()
star_r["ISO3"] = star_r["name"].apply(get_iso3)


# Coerce metrics to numeric
for c in ["STAR_R_centiSTAR", "STAR_R_mean"]:
    if c in star_r.columns:
        star_r[c] = pd.to_numeric(star_r[c], errors="coerce")

star_r = star_r[["ISO3", "STAR_R_centiSTAR", "STAR_R_mean"]].dropna(subset=["ISO3"])
star_r["ISO3"] = star_r["ISO3"].str.upper()

# Merge metrics & human-friendly country labels
merged = avg_annual.merge(star_r, on="ISO3", how="left")
merged = merged.merge(epi[["iso","country"]], left_on="ISO3", right_on="iso", how="left")
merged["country"] = merged["country"].fillna(merged["ISO3"])

# For log x-axes: keep only strictly positive STAR-R values
merged_centi = merged[["ISO3","country",f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}","STAR_R_centiSTAR"]].dropna()
merged_centi = merged_centi[merged_centi["STAR_R_centiSTAR"] > 0]
merged_mean  = merged[["ISO3","country",f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}","STAR_R_mean"]].dropna()
merged_mean  = merged_mean[merged_mean["STAR_R_mean"] > 0]


# Plot helpers

def scatter_by_act(ax, data_with_iso3, xcol, ycol, annotate=False, s=90, label_fs=11):
    tmp = data_with_iso3.merge(act_class, on="ISO3", how="left")
    tmp["act_key"] = tmp["act_key"].fillna("Act_Undef")
    for key in act_merged_order:
        sub = tmp[tmp["act_key"] == key][[xcol, ycol, "country"]].dropna()
        if not sub.empty:
            ax.scatter(sub[xcol], sub[ycol], s=s, color=color_by_key[key], alpha=0.9)
            if annotate:
                # slight right offset for readability
                for _, r in sub.iterrows():
                    ax.annotate(
                        r["country"],
                        (r[xcol], r[ycol]),
                        xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=label_fs,
                        ha="left",
                        va="center"
                    )

#-------------------------------------------------------------------------------------
# SI: STAR-R PER GOAL CLASSES
#-------------------------------------------------------------------------------------

# === Helper to build merged dataset for a given activity-share column ===
def build_activity_panel(share_col: str, out_suffix: str):
    if share_col not in df_merged_act.columns:
        raise RuntimeError(f"Expected column '{share_col}' not found in df_merged_act.")
    _annual = (
        df_merged_act
        .groupby(["ISO3", "Year"], as_index=False)[[share_col]]
        .sum()
    )
    _avg = (
        _annual
        .groupby("ISO3", as_index=False)[share_col]
        .mean()
        .rename(columns={share_col: f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_{out_suffix}"})
    )
    _merged = _avg.merge(star_r[["ISO3", "STAR_R_centiSTAR"]], on="ISO3", how="left")
    _merged = _merged.merge(epi[["iso", "country"]], left_on="ISO3", right_on="iso", how="left")
    _merged["country"] = _merged["country"].fillna(_merged["ISO3"])
    _merged = _merged[["ISO3", "country", f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_{out_suffix}", "STAR_R_centiSTAR"]].dropna()
    _merged = _merged[_merged["STAR_R_centiSTAR"] > 0]
    return _merged

def shade_quadrants(ax, x_split, y_split,
                    c_br="#f08a8f",  # bottom-right: light red
                    c_tr="#256628",  # top-right: dark green
                    c_bl="#c8e6c9",  # bottom-left: light green
                    c_tl="#bdbdbd",  # top-left: grey
                    alpha=0.15):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.add_patch(Rectangle((xmin, ymin), x_split - xmin, y_split - ymin, facecolor=c_bl, alpha=alpha, zorder=0))
    ax.add_patch(Rectangle((x_split, ymin), xmax - x_split, y_split - ymin, facecolor=c_br, alpha=alpha, zorder=0))
    ax.add_patch(Rectangle((xmin, y_split), x_split - xmin, ymax - y_split, facecolor=c_tl, alpha=alpha, zorder=0))
    ax.add_patch(Rectangle((x_split, y_split), xmax - x_split, ymax - y_split, facecolor=c_tr, alpha=alpha, zorder=0))

# == Build per-activity datasets (reusing existing PR logic) ==
merged_centi_pr   = build_activity_panel("Act_Protect_Resto_share", "PR")
merged_centi_sust = build_activity_panel("Act_SustResMgmt_share", "SUST")
merged_centi_inva = build_activity_panel("Act_Invasiv_share", "INVASIV")
merged_centi_poll = build_activity_panel("Act_Pollut_share", "POLLUT")


def add_linear_fit(ax, x, y, color="grey", linewidth=2, linestyle="-"):
    """
    Fits and plots a linear regression line (y vs x) on the current axis.
    Works with log-scaled axes by fitting to log10(x) and log10(y).
    """
    # Clean NaNs and non-positive values (since log-scale)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_fit = np.log10(x[mask])
    y_fit = np.log10(y[mask])

    if len(x_fit) < 2:
        return  # not enough data to fit

    # Fit line in log–log space
    coefs = Polynomial.fit(x_fit, y_fit, deg=1).convert().coef
    slope, intercept = coefs[1], coefs[0]

    # Build fitted line
    xx = np.linspace(x_fit.min(), x_fit.max(), 100)
    yy = intercept + slope * xx

    # Transform back to linear scale
    ax.plot(10**xx, 10**yy, color=color, linewidth=linewidth, linestyle=linestyle,
            alpha=0.8, zorder=2)

# --- 1) TOTALS: single chart and save as its own PNG ---
fig_total = plt.figure(figsize=(11, 10), constrained_layout=True)
ax_total = fig_total.add_subplot(1, 1, 1)

# Points
# Points
ax_total.scatter(
    merged_centi["STAR_R_centiSTAR"],
    merged_centi[f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}"],
    s=90, color="#555555", alpha=0.9, edgecolor="none"
)
ax_total.scatter(
    merged_centi["STAR_R_centiSTAR"],
    merged_centi[f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}"],
    s=90, facecolors="none", edgecolors="#424242", linewidths=0.8
)

# >>> Add linear fit line here <<<
add_linear_fit(
    ax_total,
    merged_centi["STAR_R_centiSTAR"].values,
    merged_centi[f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}"].values,
    color="grey", linewidth=2
)


# Do not label these ISO3 codes
skip_iso3 = [
    "MDG","HTI","TZA","UGA","TUN","PRY","SYR","ARM","MNE","LSO","ERI",
    "TCD","GIN","BDI","ZAF","GTM","MMR","HND","NPL"
]

lab1 = merged_centi[~merged_centi["ISO3"].isin(skip_iso3)]
for _, r in lab1.iterrows():
    ax_total.annotate(
        r["ISO3"],
        (r["STAR_R_centiSTAR"], r[f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}"]),
        xytext=(0, 5), textcoords="offset points",
         fontsize=10.5, ha="center", va="bottom"
    )


# Scales, labels
ax_total.set_xscale("log"); ax_total.set_yscale("log")
ax_total.set_xlabel("log STAR-R (centi)", fontsize=16)
ax_total.set_ylabel(f"log Average annual disbursement (USD mn, {start_year_of_interest}-{end_year_of_interest})", fontsize=16)
#ax_total.set_title("Total disbursement", fontsize=15, pad=10)
ax_total.tick_params(axis="both", labelsize=12)
ax_total.grid(False)

# Limits + medians for shading/split (these will also feed the 4-panel figure)
xlim_total = ax_total.get_xlim()
y_nonzero_total = merged_centi[f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}"][merged_centi[f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}"] > 0]
yq_total = float(np.median(y_nonzero_total))
x_split_total = float(np.median(merged_centi["STAR_R_centiSTAR"].values))
ax_total.set_xlim(xlim_total)
ax_total.set_ylim(y_nonzero_total.min() * 0.8, y_nonzero_total.max() * 1.2)

shade_quadrants(ax_total, x_split_total, yq_total)
ax_total.axvline(x_split_total, linestyle="--", linewidth=1, color="#424242")
ax_total.axhline(yq_total, linestyle="--", linewidth=1, color="#424242")

# Save single PNG
out_total = "outputs/MAIN_Star_R_Scatterplot.png"
plt.savefig(out_total, dpi=300, bbox_inches="tight")
plt.savefig("outputs/MAIN_Star_R_Scatterplot.pdf", dpi=300, bbox_inches="tight")
plt.show()
print(f"Main Figure: STAR-R Scatterplot saved to: {out_total} and .pdf")


# --- 2) FOUR CATEGORY PANELS together with unified y-scaling (ymax from TOTAL) ---
fig4 = plt.figure(figsize=(24, 8), constrained_layout=True)
gs4 = gridspec.GridSpec(2, 4, height_ratios=[14, 1], figure=fig4)
gs4.update(wspace=0.07)  # increase horizontal space between the 4 plots


ax2 = fig4.add_subplot(gs4[0, 0])  # PR
ax3 = fig4.add_subplot(gs4[0, 1])  # SUST
ax4 = fig4.add_subplot(gs4[0, 2])  # INVASIV
ax5 = fig4.add_subplot(gs4[0, 3])  # POLLUT

# Bottom spacer row (no legend)
for j in range(0, 4):
    _ax = fig4.add_subplot(gs4[1, j]); _ax.axis("off")

# Titles
ax2.set_title("Protection, conservation\nand restoration\n", fontsize=18, fontweight='bold', pad=10)
ax3.set_title("Sustainable resource\nmanagement\n", fontsize=18, fontweight='bold', pad=10)
ax4.set_title("Invasive species\nmanagement\n", fontsize=18, fontweight='bold', pad=10)
ax5.set_title("Pollution\ncontrol\n", fontsize=18, fontweight='bold',  pad=10)


# --- Define which countries to label per category ---
labels_PR = [
    "TKM","LSO","LBY","SUR","DJI","JOR","GUY","PRK","AZE","SLE","IRN","VEN","UKR",
    "THA","ETH","PHL","CRI","CHN","IND","COL","BOL","MOZ","NAM","SEN","MRT","LAO",
    "PER","ARG"
]

labels_SUST = [
    "DJI","SUR","JOR","TJK","MRT","SLV","GUY","LBY","SWZ","TKM","MNE","BLR","AZE",
    "TGO","DZA","MYS","THA","PAN","DZA","AGO","NER","BGD","VNM","COL","CHN","BRA",
    "ETH","PHL","CRI","MEX","UKR"
]

labels_INVASIV = [
    "DJI","JOR","GMB","ALB","LBN","LAO","YEM","BWA","NAM","VEN","THA","CRI","IRO",
    "VUT","ZMB","BFA","EGY","SOM","NGA","KEN","BRA","TZA","CHN","PHL","MEX","IND",
    "VNM","COL","HND","MOZ","BDI"
]

labels_POLLUT = [
    "JOR","DJI","SUR","LBY","GUY","SWZ","TJK","SEN","BWA","NAM","BLZ","KAZ","PRK",
    "DZA","MYS","VEN","THA","MEX","JAM","CRI","PHL","ETH","BRA","CHN","IND","VNM",
    "EGY","BGD","MOZ","COL","NGA","SYR","ZMB","ECU"]


# Helper: draw one category panel
def draw_category_panel(ax, df_cat, ycol, color_key, ylab, fontname="Arial"):
    ax.scatter(
        df_cat["STAR_R_centiSTAR"], df_cat[ycol],
        s=90, alpha=0.9, edgecolor="none", color=color_by_key[color_key]
    )
    ax.scatter(
        df_cat["STAR_R_centiSTAR"], df_cat[ycol],
        s=90, facecolors="none", edgecolors="#424242", linewidths=0.8
    )

    # >>> Add linear fit line here <<<
    add_linear_fit(ax, df_cat["STAR_R_centiSTAR"].values,
                        df_cat[ycol].values,
                        color="grey", linewidth=2)

    # Labels and scales
    # Choose the right label list for this panel
    if color_key == "Act_Protect_Resto":
        allowed_labels = labels_PR
    elif color_key == "Act_SustResMgmt":
        allowed_labels = labels_SUST
    elif color_key == "Act_Invasiv":
        allowed_labels = labels_INVASIV
    elif color_key == "Act_Pollut":
        allowed_labels = labels_POLLUT
    else:
        allowed_labels = []

    # Annotate only allowed ISO3s
    for _, r in df_cat.iterrows():
        if r["ISO3"] in allowed_labels:
            ax.annotate(
                r["ISO3"],
                (r["STAR_R_centiSTAR"], r[ycol]),
                xytext=(0, 5), textcoords="offset points",
                fontname=fontname, fontsize=12, ha="center", va="bottom"
            )

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("log STAR-R (centi)", fontsize=16)
    ax.set_ylabel(ylab, fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(xlim_total)

# Draw all 4
ycol_pr   = f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_PR"
ycol_sust = f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_SUST"
ycol_inva = f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_INVASIV"
ycol_poll = f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_POLLUT"

draw_category_panel(
    ax2, merged_centi_pr, ycol_pr, "Act_Protect_Resto",
    f"log Average annual disbursement (USD mn, {start_year_of_interest}-{end_year_of_interest})"
)
draw_category_panel(
    ax3, merged_centi_sust, ycol_sust, "Act_SustResMgmt",
    f"log Average annual disbursement (USD mn, {start_year_of_interest}-{end_year_of_interest})"
)
draw_category_panel(
    ax4, merged_centi_inva, ycol_inva, "Act_Invasiv",
    f"log Average annual disbursement (USD mn, {start_year_of_interest}-{end_year_of_interest})"
)
draw_category_panel(
    ax5, merged_centi_poll, ycol_poll, "Act_Pollut",
    f"log Average annual disbursement (USD mn, {start_year_of_interest}-{end_year_of_interest})"
)

# Remove y-axis labels & tick labels for second, third, fourth charts
for ax in (ax3, ax4, ax5):
    ax.set_ylabel("")

# Initial y-lims for each (based on their own positive values)
def _pos_ylim(df, col):
    _y = df[col][df[col] > 0]
    return (_y.min() * 0.8, _y.max() * 1.2)

yl_pr   = _pos_ylim(merged_centi_pr,   ycol_pr)
yl_sust = _pos_ylim(merged_centi_sust, ycol_sust)
yl_inva = _pos_ylim(merged_centi_inva, ycol_inva)
yl_poll = _pos_ylim(merged_centi_poll, ycol_poll)

ax2.set_ylim(*yl_pr)
ax3.set_ylim(*yl_sust)
ax4.set_ylim(*yl_inva)
ax5.set_ylim(*yl_poll)

# --- Harmonize y-scale across the four category panels
# Rule (same as before): ymin = min across the 4 panels; ymax = max from TOTAL panel
ymins = [ax2.get_ylim()[0], ax3.get_ylim()[0], ax4.get_ylim()[0], ax5.get_ylim()[0]]
ymin_4 = min(ymins)
ymax_4 = max(ax2.get_ylim()[1], ax3.get_ylim()[1], ax4.get_ylim()[1], ax5.get_ylim()[1])
for ax in (ax2, ax3, ax4, ax5):
    ax.set_ylim(ymin_4, ymax_4)

# Median splits for each category (for shading)
def _median_xy(df, ycol):
    ypos = df[ycol][df[ycol] > 0]
    return float(np.median(df["STAR_R_centiSTAR"].values)), float(np.median(ypos))

x_split_right_pr,   yq_right_pr   = _median_xy(merged_centi_pr,   ycol_pr)
x_split_right_sust, yq_right_sust = _median_xy(merged_centi_sust, ycol_sust)
x_split_right_inva, yq_right_inva = _median_xy(merged_centi_inva, ycol_inva)
x_split_right_poll, yq_right_poll = _median_xy(merged_centi_poll, ycol_poll)

# Shading + split lines on category panels
for (ax, x_split, yq) in [
    (ax2, x_split_right_pr,   yq_right_pr),
    (ax3, x_split_right_sust, yq_right_sust),
    (ax4, x_split_right_inva, yq_right_inva),
    (ax5, x_split_right_poll, yq_right_poll),
]:
    shade_quadrants(ax, x_split, yq)
    ax.axvline(x_split, linestyle="--", linewidth=1, color="#424242")
    ax.axhline(yq,      linestyle="--", linewidth=1, color="#424242")

# Save 4-panel PNG
out_four = "outputs/SI_Star_R_Scatterplot_goal_classes.png"
plt.savefig(out_four, dpi=300, bbox_inches="tight")
plt.savefig("outputs/SI_Star_R_Scatterplot_goal_classes.pdf", dpi=300, bbox_inches="tight")
plt.show()
print(f"SI Figure: Detailed STAR-R scatterplots saved to: {out_four} and .pdf")


# _______________________________________
# TABLE Regression export (XLSX)
# _______________________________________

try:
    from scipy import stats as st
    _has_scipy = True
except Exception:
    _has_scipy = False

os.makedirs(
    "./outputs", exist_ok=True)

# --- Helper: prepare X, y (transform controlled by 'transform' arg) ---
def _prep_xy(df: pd.DataFrame, ycol: str, xcol: str = "STAR_R_centiSTAR",
             transform: str = "log10") -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    cols = ["ISO3", xcol, ycol]
    _df = df.loc[:, cols].dropna()
    if transform == "log10":
        _df = _df[(_df[xcol] > 0) & (_df[ycol] > 0)]
        X = np.log10(_df[xcol])
        y = np.log10(_df[ycol])
    elif transform == "identity":
        X = _df[xcol]
        y = _df[ycol]
    elif transform == "log10_y_only":
        _df = _df[_df[ycol] > 0]
        X = _df[xcol]
        y = np.log10(_df[ycol])
    elif transform == "log10_x_only":
        _df = _df[_df[xcol] > 0]
        X = np.log10(_df[xcol])
        y = _df[ycol]
    else:
        raise ValueError("Unknown transform")

    X = pd.Series(X, name=xcol)
    y = pd.Series(y, name=ycol)
    return X, y, _df

# --- Helper: run OLS with classic and robust SEs; compute correlations ---
def run_ols_report(df: pd.DataFrame, ycol: str, transform: str = "log10") -> Dict:
    X, y, clean = _prep_xy(df, ycol=ycol, transform=transform)

    # Align indices and keep labeled columns
    idx = X.index.intersection(y.index)
    X = X.loc[idx]
    y = y.loc[idx]
    X_df = X.to_frame()
    X_const = sm.add_constant(X_df, has_constant='add')

    model = sm.OLS(y, X_const, missing="drop").fit()
    model_hc1 = model.get_robustcov_results(cov_type="HC1")
    xname = X.name

    # ---- helpers that work for pandas- or ndarray-backed results ----
    def _names(res):
        try:
            return list(res.params.index)
        except Exception:
            return list(res.model.exog_names)

    def _get_param(res, name):
        p = res.params
        if hasattr(p, "get"):
            return float(p.get(name, np.nan))
        names = _names(res)
        j = names.index(name)
        return float(p[j])

    def _get_pval(res, name):
        p = res.pvalues
        if hasattr(p, "get"):
            return float(p.get(name, np.nan))
        names = _names(res)
        j = names.index(name)
        return float(p[j])

    def _get_ci(res, name):
        ci = res.conf_int()
        if isinstance(ci, np.ndarray):
            names = _names(res)
            j = names.index(name)
            return float(ci[j, 0]), float(ci[j, 1])
        vals = ci.loc[name].tolist()
        return float(vals[0]), float(vals[1])

    # ---------------- core extracts ----------------
    slope     = _get_param(model, xname)
    intercept = _get_param(model, "const")
    pval      = _get_pval(model, xname)
    ci_low, ci_high = _get_ci(model, xname)

    slope_rb  = _get_param(model_hc1, xname)
    pval_rb   = _get_pval(model_hc1, xname)
    ci_low_rb, ci_high_rb = _get_ci(model_hc1, xname)

    r2 = float(model.rsquared)
    n  = int(model.nobs)

    # Correlations on the variables actually used in the regression
    pearson_r = float(pd.Series(X).corr(pd.Series(y)))
    pearson_p = np.nan
    spearman_r = float(pd.Series(X).rank().corr(pd.Series(y).rank(), method="pearson"))
    spearman_p = np.nan
    if _has_scipy:
        pearson_r, pearson_p = st.pearsonr(X, y)
        spearman_r, spearman_p = st.spearmanr(X, y)

    return {
        "N": n,
        "Transform": transform,
        "Slope (β1)": slope,
        "Intercept (β0)": intercept,
        "p-value (β1)": pval,
        "95% CI low": ci_low,
        "95% CI high": ci_high,
        "R-squared": r2,
        "Slope robust (HC1)": slope_rb,
        "p-value robust (HC1)": pval_rb,
        "95% CI low robust (HC1)": ci_low_rb,
        "95% CI high robust (HC1)": ci_high_rb,
        "Pearson r": float(pearson_r),
        "Pearson p": float(pearson_p) if not np.isnan(pearson_p) else np.nan,
        "Spearman ρ (ranks)": float(spearman_r),
        "Spearman p": float(spearman_p) if not np.isnan(spearman_p) else np.nan,
    }


# --- Define the five panels (df, y-column, label).
PANELS = [
    ("Total disbursement",        merged_centi,        f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}",            "identity"),
    ("Protection/Restoration",    merged_centi_pr,     f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_PR",         "identity"),
    ("Sust. resource mgmt.",      merged_centi_sust,   f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_SUST",       "identity"),
    ("Invasive species mgmt.",    merged_centi_inva,   f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_INVASIV",    "identity"),
    ("Pollution control",         merged_centi_poll,   f"Avg_Annual_Disb_{start_year_of_interest}_{end_year_of_interest}_POLLUT",     "identity"),
]

# --- Run and collect ---
rows = []
details: Dict[str, pd.DataFrame] = {}
for name, dfp, ycol, transform in PANELS:
    res = run_ols_report(dfp, ycol=ycol, transform=transform)
    res_row = {"Panel": name, "y": ycol, **res}
    rows.append(res_row)

    # Store per-observation data (X, y, fitted, resid) for diagnostics
    X, y, clean = _prep_xy(dfp, ycol=ycol, transform=transform)
    idx = X.index.intersection(y.index)
    X = X.loc[idx]
    y = y.loc[idx]
    Xc = sm.add_constant(X.to_frame(), has_constant='add')
    fitted = sm.OLS(y, Xc).fit().fittedvalues
    resid = y - fitted

    if transform == "identity":
        x_label = "X_STAR_R"
        y_label = "Y"
        fit_label = "Fitted_Y"
    else:
        x_label = "log10_X_STAR_R"
        y_label = "log10_Y"
        fit_label = "Fitted_log10_Y"

    details[name] = pd.DataFrame({
        "ISO3": clean.loc[idx, "ISO3"].values,
        x_label: X.values,
        y_label: y.values,
        fit_label: fitted.values,
        "Residuals": resid.values
    })

summary_df = pd.DataFrame(rows)

# --- Helper: Excel-safe, unique sheet names (≤31 chars, no invalid chars) ---
def _sanitize_sheet_name(name: str) -> str:
    invalid = '[]:*?/\\'
    s = ''.join('_' if c in invalid else c for c in name).strip()
    return s or "Sheet"

def _unique_sheet_name(base: str, used: set) -> str:
    base = _sanitize_sheet_name(base)
    base = base[:31]
    name = base or "Sheet"
    i = 1
    while name in used:
        suffix = f" ({i})"
        name = (base[:31 - len(suffix)] or "Sheet") + suffix
        i += 1
    used.add(name)
    return name

# --- Write to Excel (one summary sheet + one sheet per panel with diagnostics) ---
out_xlsx = "outputs/SI_STAR_R_regression_table.xlsx"
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xlw:
    # Summary
    summary_cols = [
        "Panel", "y", "N", "Transform",
        "Slope (β1)", "p-value (β1)", "95% CI low", "95% CI high", "R-squared",
        "Slope robust (HC1)", "p-value robust (HC1)", "95% CI low robust (HC1)", "95% CI high robust (HC1)",
        "Pearson r", "Pearson p", "Spearman ρ (ranks)", "Spearman p",
    ]
    summary_df.loc[:, summary_cols].to_excel(xlw, sheet_name="SUMMARY", index=False)

    # Per panel diagnostics (sanitize & dedupe sheet names)
    used_names = {"SUMMARY"}
    for name, df_diag in details.items():
        sheet = _unique_sheet_name(name, used_names)
        df_diag.to_excel(xlw, sheet_name=sheet, index=False)

print(f"SI Table: Exported regression results to: {out_xlsx}")


# ---------------------------------------
# CONFIGURATION WORLD MAPs IMPLEMENTATION
# ---------------------------------------


# Adjust colorbar size to fit within each row (per map)
def get_colorbar_settings(col_index, row_index, total_rows):
    domain_height = 1 / total_rows
    return dict(
        x=0.46 if col_index == 1 else 1.04,
        y=1 - (row_index - 0.5) * domain_height,
        len=domain_height * 0.88,  # ~88% of map height (excludes title space)
        thickness=10,
        tickfont=dict(color='black'),
        title=dict(text="USD mn", font=dict(color='black'))
    )

def standard_geo_layout():
    return dict(
        showframe=True,
        showcoastlines=True,
        showland=True,
        landcolor='white',
        lakecolor='white',
        coastlinecolor='black',
        projection_type='equirectangular'
    )


# Implementation columns and their colors
impl_labels = {
    'Impl_Infra': 'Infrastructure',
    'Impl_Know': 'Awareness and knowledge',
    'Impl_Regul': 'Policy, regulation and governance',
    'Impl_Undef': 'Undefined'
}

# Prepare a filtered copy for mapping
df_map = df[df['Year'].isin(years_of_interest)].copy()

# Clean disbursements
df_map['USD_Disbursement'] = pd.to_numeric(df_map['USD_Disbursement'], errors='coerce').fillna(0)
df_map['USD_Disbursement'] = df_map['USD_Disbursement'] * 1000  # From Bio → Mio
df_map.loc[df_map['USD_Disbursement'] < 0, 'USD_Disbursement'] = 0
df_map[impl_order] = df_map[impl_order].fillna(0).astype(int)

# Keep only rows with implementation activity
df_map['impl_count'] = df_map[impl_order].sum(axis=1)
df_map = df_map[df_map['impl_count'] > 0]

# Split disbursements proportionally
for col in impl_order:
    df_map[f'{col}_share'] = df_map[col] * df_map['USD_Disbursement'] / df_map['impl_count']

# ISO3 conversion
def get_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

df_map['ISO3'] = df_map['RecipientName'].apply(get_iso3)


# Compute average disbursement per ISO3 and implementation class
results = {}
max_per_impl = {}

for col in impl_order:
    temp = df_map.groupby(['ISO3', 'Year'])[f'{col}_share'].sum().reset_index()
    annual_avg = temp.groupby('ISO3')[f'{col}_share'].mean().reset_index()
    annual_avg.rename(columns={f'{col}_share': 'avg_annual_disbursement'}, inplace=True)
    results[col] = annual_avg
    max_per_impl[col] = annual_avg['avg_annual_disbursement'].max()

# Create 2x2 subplot grid
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f"<b>{impl_labels[col]}</b>" for col in impl_order],
    specs=[[{"type": "choropleth"}, {"type": "choropleth"}],
           [{"type": "choropleth"}, {"type": "choropleth"}]],
    horizontal_spacing=0.15,
    vertical_spacing=0.08
)

impl_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
colorbar_positions = [
    get_colorbar_settings(c, r, total_rows=2)
    for (r, c) in impl_positions
]

# Add choropleth maps
for (col_name, color), (row, col_pos), cbar in zip(zip(impl_order, impl_colors), impl_positions, colorbar_positions):
    data = results[col_name]
    zmax = max_per_impl[col_name]

    fig.add_trace(
        go.Choropleth(
            locations=data["ISO3"],
            z=data["avg_annual_disbursement"],
            text=data["ISO3"],
            colorscale=[[0.0, 'white'], [1.0, color]],
            zmin=0,
            zmax=zmax,
            colorbar=cbar,
            showscale=True,
            geo=f'geo{(row - 1) * 2 + col_pos}'
        ),
        row=row, col=col_pos
    )

# ---------------------------------------
# CONFIGURATION WORLD MAPs GOAL
# ---------------------------------------

# Prepare data
df_act = df[df['Year'].isin(years_of_interest)].copy()
df_act['USD_Disbursement'] = pd.to_numeric(df_act['USD_Disbursement'], errors='coerce').fillna(0)
df_act.loc[df_act['USD_Disbursement'] < 0, 'USD_Disbursement'] = 0
df_act['USD_Disbursement'] = df_act['USD_Disbursement'] * 1000
df_act[act_order] = df_act[act_order].fillna(0).astype(int)
df_act['act_count'] = df_act[act_order].sum(axis=1)
df_act = df_act[df_act['act_count'] > 0]


# Split disbursements
for col in act_order:
    df_act[f'{col}_share'] = df_act[col] * df_act['USD_Disbursement'] / df_act['act_count']

df_act['ISO3'] = df_act['RecipientName'].apply(get_iso3)

# Aggregate
act_results = {}
max_per_act = {}
for col in act_order:
    temp = df_act.groupby(['ISO3', 'Year'])[f'{col}_share'].sum().reset_index()
    avg = temp.groupby('ISO3')[f'{col}_share'].mean().reset_index()
    avg.rename(columns={f'{col}_share': 'avg_annual_disbursement'}, inplace=True)
    act_results[col] = avg
    max_per_act[col] = avg['avg_annual_disbursement'].max()

# Create subplot grid
num_act_maps = len(act_order)
rows_act = (num_act_maps + 1) // 2

fig_act = make_subplots(
    rows=rows_act, cols=2,
    subplot_titles=[f"<b>{act_labels[col]}</b>" for col in act_order],
    specs=[[{"type": "choropleth"}, {"type": "choropleth"}]] * rows_act,
    horizontal_spacing=0.15,
    vertical_spacing=0.08
)

# Positions and colorbars
act_positions = [(r, c) for r in range(1, rows_act+1) for c in range(1, 3)]
colorbars = [
    get_colorbar_settings(c, r, rows_act)
    for (r, c) in act_positions
]

# Add maps
for (col_name, color), (row, col_pos), cbar in zip(zip(act_order, act_colors), act_positions, colorbars):
    data = act_results[col_name]
    zmax = max_per_act[col_name]

    fig_act.add_trace(
        go.Choropleth(
            locations=data["ISO3"],
            z=data["avg_annual_disbursement"],
            text=data["ISO3"],
            colorscale=[[0.0, 'white'], [1.0, color]],
            zmin=0,
            zmax=zmax,
            colorbar=cbar,
            showscale=True,
            geo=f'geo{(row - 1) * 2 + col_pos}'
        ),
        row=row, col=col_pos
    )

# Layout
fig_act.update_layout(
    height=450 * rows_act,
    width=1200,
    title=dict(
        text=f"<b>Annual disbursement by goal ({start_year_of_interest}-{end_year_of_interest} average)</b>",
        x=0.5,
        font=dict(size=20, color='black')
    ),
    margin=dict(t=90, b=30, l=20, r=30),
    showlegend=False,
    font=dict(color='black')
)

fig_act.update_layout(**{f'geo{i}': standard_geo_layout()})


# Tweak annotations
for annotation in fig_act['layout']['annotations']:
    annotation['yshift'] = -20


# Apply uniform geos
for i in range(1, num_act_maps + 1):
    fig_act.update_layout(**{
        f'geo{i}': dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    })


# ---------------------------------------
# CONFIGURATION WORLD MAPs ECOSYSTEM TYPE
# ---------------------------------------

df_eco = df[df['Year'].isin(years_of_interest)].copy()
df_eco['USD_Disbursement'] = pd.to_numeric(df_eco['USD_Disbursement'], errors='coerce').fillna(0)
df_eco.loc[df_eco['USD_Disbursement'] < 0, 'USD_Disbursement'] = 0
df_eco['USD_Disbursement'] = df_eco['USD_Disbursement'] * 1000  # From Bio → Mio
df_eco[eco_order] = df_eco[eco_order].fillna(0).astype(int)
df_eco['eco_count'] = df_eco[eco_order].sum(axis=1)
df_eco = df_eco[df_eco['eco_count'] > 0]

for col in eco_order:
    df_eco[f'{col}_share'] = df_eco[col] * df_eco['USD_Disbursement'] / df_eco['eco_count']

df_eco['ISO3'] = df_eco['RecipientName'].apply(get_iso3)

eco_results = {}
max_per_eco = {}
for col in eco_order:
    temp = df_eco.groupby(['ISO3', 'Year'])[f'{col}_share'].sum().reset_index()
    avg = temp.groupby('ISO3')[f'{col}_share'].mean().reset_index()
    avg.rename(columns={f'{col}_share': 'avg_annual_disbursement'}, inplace=True)
    eco_results[col] = avg
    max_per_eco[col] = avg['avg_annual_disbursement'].max()

num_eco_maps = len(eco_order)
rows_eco = (num_eco_maps + 1) // 2

fig_eco = make_subplots(
    rows=rows_eco, cols=2,
    subplot_titles=[f"<b>{eco_labels[col]}</b>" for col in eco_order],
    specs=[[{"type": "choropleth"}, {"type": "choropleth"}]] * rows_eco,
    horizontal_spacing=0.15,
    vertical_spacing=0.08
)

eco_positions = [(r, c) for r in range(1, rows_eco+1) for c in range(1, 3)]
colorbars = [
    get_colorbar_settings(c, r, rows_eco)
    for (r, c) in eco_positions
]

for (col_name, color), (row, col_pos), cbar in zip(zip(eco_order, eco_colors), eco_positions, colorbars):
    data = eco_results[col_name]
    zmax = max_per_eco[col_name]

    fig_eco.add_trace(
        go.Choropleth(
            locations=data["ISO3"],
            z=data["avg_annual_disbursement"],
            text=data["ISO3"],
            colorscale=[[0.0, 'white'], [1.0, color]],
            zmin=0,
            zmax=zmax,
            colorbar=cbar,
            showscale=True,
            geo=f'geo{(row - 1) * 2 + col_pos}'
        ),
        row=row, col=col_pos
    )

# ---------------------------------------
# TABLE COUNTRY CATEGORY EXCEL SHARES XLSX
# ---------------------------------------
YEARS = years_of_interest

# Fallbacks if *_order not defined
_act_order  = act_order  if 'act_order'  in globals() else act_cols
_impl_order = impl_order if 'impl_order' in globals() else impl_cols
_eco_order  = eco_order  if 'eco_order'  in globals() else eco_cols


def _prep_dimension(base_df: pd.DataFrame, subclass_order):

    df_dim = base_df[base_df['Year'].isin(YEARS)].copy()

    # Clean disbursements (exactly like maps)
    df_dim['USD_Disbursement'] = pd.to_numeric(
        df_dim['USD_Disbursement'], errors='coerce'
    ).fillna(0)
    df_dim.loc[df_dim['USD_Disbursement'] < 0, 'USD_Disbursement'] = 0
    df_dim['USD_Disbursement'] = df_dim['USD_Disbursement'] * 1000  # Bio → Mn

    # Ensure flags exist, fillna 0, cast to int (like maps)
    for c in subclass_order:
        if c not in df_dim.columns:
            df_dim[c] = 0
    df_dim[subclass_order] = df_dim[subclass_order].fillna(0).astype(int)

    # Keep only rows with activity in this dimension
    count_col = '_tmp_count'
    df_dim[count_col] = df_dim[subclass_order].sum(axis=1)
    df_dim = df_dim[df_dim[count_col] > 0].copy()

    # Proportional split (per-row share)
    for col in subclass_order:
        df_dim[f'{col}_share'] = (
            df_dim[col] * df_dim['USD_Disbursement'] / df_dim[count_col]
        )

    return df_dim


def _country_level_annual_avg(df_dim: pd.DataFrame, subclass_order):
    """Sum by (RecipientName, Year), then mean over years (maps logic)."""
    rows = []
    for col in subclass_order:
        temp = (
            df_dim
            .groupby(['RecipientName', 'Year'], dropna=False)[f'{col}_share']
            .sum()
            .reset_index()
        )
        # Mean across whatever years are present for that country (no reindexing to zeros)
        avg = (
            temp
            .groupby('RecipientName', dropna=False)[f'{col}_share']
            .mean()
            .rename(col)
            .reset_index()
        )
        rows.append(avg)

    # Wide by RecipientName
    out = rows[0]
    for part in rows[1:]:
        out = out.merge(part, on='RecipientName', how='outer')

    # ISO3 (optional helper column)
    if 'ISO3' in df.columns:
        iso_map = df[['RecipientName', 'ISO3']].dropna().drop_duplicates()
    else:
        # Try to compute ISO3 like in maps
        try:
            out_iso = df[['RecipientName']].drop_duplicates().copy()
            out_iso['ISO3'] = out_iso['RecipientName'].apply(get_iso3)
            iso_map = out_iso
        except Exception:
            iso_map = pd.DataFrame({'RecipientName': out['RecipientName'], 'ISO3': np.nan})

    out = out.merge(iso_map, on='RecipientName', how='left')

    # Order columns
    first_cols = ['RecipientName', 'ISO3']
    subclass_cols = [c for c in subclass_order if c in out.columns]
    out = out.reindex(columns=first_cols + subclass_cols)

    # Sort + keep as float (no rounding here)
    out = out.sort_values('RecipientName').reset_index(drop=True)
    out[subclass_cols] = out[subclass_cols].astype(float)

    if 'act_labels_nobreak' in globals():
        labels = {**act_labels_nobreak, **impl_labels_nobreak, **eco_labels_nobreak}
        out = out.rename(columns={c: labels.get(c, c) for c in subclass_cols})

    # Friendly column name for country
    out = out.rename(columns={'RecipientName': 'Country'})
    return out


# Build three tables using the exact map logic
df_impl_base = _prep_dimension(df, _impl_order)
impl_table   = _country_level_annual_avg(df_impl_base, _impl_order)

df_act_base  = _prep_dimension(df, _act_order)
act_table    = _country_level_annual_avg(df_act_base, _act_order)

df_eco_base  = _prep_dimension(df, _eco_order)
eco_table    = _country_level_annual_avg(df_eco_base, _eco_order)


# ---------------------------------------
# Extra columns for Act / Impl / Eco
# ---------------------------------------

def _safe_pct(df, num_col, denom_col, new_col):
    """Generic safe percentage helper (used for Impl / Eco)."""
    df[new_col] = np.where(
        df[denom_col] > 0,
        df[num_col] / df[denom_col],
        0.0,
    )


# ---------- Act sheet ----------
act_base_cols = [
    "Protection and conservation",
    "Restoration",
    "Sustainable agriculture",
    "Sustainable forest management",
    "Sustainable fishery",
    "Sustainable water management",
    "Sustainable mgmt. of other natural resources",
    "Invasive species management",
    "Pollution control",
    "Undefined",
]

# total including undefined (absolute amounts)
act_table["Total"] = act_table[act_base_cols].sum(axis=1)

# total excluding undefined
act_wo_undef_cols = [c for c in act_base_cols if c != "Undefined"]
act_table["Total w/o undefined"] = act_table[act_wo_undef_cols].sum(axis=1)

# % of Total (each country's Total in the overall Total, across all countries)
total_all_countries = act_table["Total"].sum()
act_table["% of Total"] = np.where(
    total_all_countries > 0,
    act_table["Total"] / total_all_countries,
    0.0,
)

# groupings in absolute values
act_table["Protection, conservation & restoration"] = (
    act_table["Protection and conservation"] + act_table["Restoration"]
)

act_table["Sustainable Management"] = act_table[
    [
        "Sustainable agriculture",
        "Sustainable forest management",
        "Sustainable fishery",
        "Sustainable water management",
        "Sustainable mgmt. of other natural resources",
    ]
].sum(axis=1)

# final shares for the 5 Act groups:
# Undefined, Invasive species, Pollution control,
# Protection, conservation & restoration (Protection + Restoration),
# Sustainable Management (sum of the 5 sustainable columns)
act_table["%_Undefined"] = np.where(
    act_table["Total"] > 0,
    act_table["Undefined"] / act_table["Total"],
    0.0,
)
act_table["%_Invasive species"] = np.where(
    act_table["Total"] > 0,
    act_table["Invasive species management"] / act_table["Total"],
    0.0,
)
act_table["%_Pollution control"] = np.where(
    act_table["Total"] > 0,
    act_table["Pollution control"] / act_table["Total"],
    0.0,
)
act_table["%_Protection, conservation & restoration"] = np.where(
    act_table["Total"] > 0,
    (
        act_table["Protection and conservation"]
        + act_table["Restoration"]
    )
    / act_table["Total"],
    0.0,
)
act_table["%_Sustainable Management"] = np.where(
    act_table["Total"] > 0,
    (
        act_table[
            [
                "Sustainable agriculture",
                "Sustainable forest management",
                "Sustainable fishery",
                "Sustainable water management",
                "Sustainable mgmt. of other natural resources",
            ]
        ].sum(axis=1)
    )
    / act_table["Total"],
    0.0,
)


# ---------- Impl sheet ----------
impl_base_cols = [
    "Policy, regulation and governance",
    "Awareness and knowledge",
    "Infrastructure",
    "Undefined",
]

impl_table["Total"] = impl_table[impl_base_cols].sum(axis=1)

_safe_pct(
    impl_table,
    "Policy, regulation and governance",
    "Total",
    "%_Policy, regulatory and governance",
)
_safe_pct(
    impl_table,
    "Awareness and knowledge",
    "Total",
    "%_Awareness and knowledge",
)
_safe_pct(
    impl_table,
    "Infrastructure",
    "Total",
    "%_Infrastructure building",
)
_safe_pct(
    impl_table,
    "Undefined",
    "Total",
    "%_Undefined",
)


# ---------- Eco sheet ----------
eco_base_cols = [
    "Crop-, range-, grass-, and arid land",
    "Forest",
    "Marine and freshwater",
    "Urban and industrial",
    "Undefined",
]

eco_table["Total"] = eco_table[eco_base_cols].sum(axis=1)

_safe_pct(
    eco_table,
    "Crop-, range-, grass-, and arid land",
    "Total",
    "%_Crop-, range-, grass-, and arid land",
)
_safe_pct(eco_table, "Forest", "Total", "%_Forest")
_safe_pct(eco_table, "Marine and freshwater", "Total", "%_Marine and freshwater")
_safe_pct(eco_table, "Urban and industrial", "Total", "%_Urban and industrial")
_safe_pct(eco_table, "Undefined", "Total", "%_Undefined")


# Write to Excel (openpyxl avoids extra dependency)
os.makedirs('./outputs', exist_ok=True)
xlsx_path = 'outputs/avg_annual_disbursement_by_country_and_subclass.xlsx'
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    act_table.to_excel(writer,  sheet_name='Act',  index=False)
    impl_table.to_excel(writer, sheet_name='Impl', index=False)
    eco_table.to_excel(writer,  sheet_name='Eco',  index=False)

print(f"SI Table: Annual disbursements by country saved: {xlsx_path}")


# ---------------------------------------
# SI: COMBINED WORLD MAPS ALL DIMENSIONS
# ---------------------------------------

# Max per dimension
zmax_act = max([v for k, v in max_per_act.items() if "Act_Undef" not in k])
zmax_impl = max([v for k, v in max_per_impl.items() if "Impl_Undef" not in k])
zmax_eco = max([v for k, v in max_per_eco.items() if "Eco_Undef" not in k])

# Define maps (18 maps, in row-major order)
maps = [
    # Targeted Action (rows 1–3)
    ("Act_Protect", "Protection and conservation", "#8e44ad", act_results, zmax_act, "Act"),
    ("Act_Resto", "Restoration", "#8e44ad", act_results, zmax_act, "Act"),
    ("Act_Agri", "Sustainable agriculture", "#8e44ad", act_results, zmax_act, "Act"),
    ("Act_ForestMgmt", "Sustainable forest management", "#8e44ad", act_results, zmax_act, "Act"),
    ("Act_Fish", "Sustainable fishery", "#8e44ad", act_results, zmax_act, "Act"),
    ("Act_WaterMgmt", "Sustainable water management", "#8e44ad", act_results, zmax_act, "Act"),
    ("Act_OthMgmt", "Sustainable mgmt. of other natural resources", "#8e44ad", act_results, zmax_act, "Act"),
    ("Act_Invasiv", "Invasive species management", "#8e44ad", act_results, zmax_act, "Act"),
    ("Act_Pollut", "Pollution control", "#8e44ad", act_results, zmax_act, "Act"),

    # Implementation Tools (row 4)
    ("Impl_Regul", "Policy, regulation and governance", "#aab63d", results, zmax_impl, "Impl"),
    ("Impl_Know", "Awareness and knowledge", "#aab63d", results, zmax_impl, "Impl"),
    ("Impl_Infra", "Infrastructure", "#aab63d", results, zmax_impl, "Impl"),

    # Ecosystem Types (rows 5–6)
    ("Eco_CropGrass", "Cropland, rangeland, grassland, arid land", "#6bc1a4", eco_results, zmax_eco, "Eco"),
    ("Eco_Forest", "Forest", "#6bc1a4", eco_results, zmax_eco, "Eco"),
    ("Eco_SeaWater", "Marine and freshwater", "#6bc1a4", eco_results, zmax_eco, "Eco"),
    ("Eco_UrbInd", "Urban and industrial", "#6bc1a4", eco_results, zmax_eco, "Eco"),
]

# Grid layout: 6 rows × 3 columns
ncols = 3
nrows = (len(maps) + ncols - 1) // ncols

# Setup figure
fig = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=[f"<b>{label}</b>" for _, label, *_ in maps],
    specs=[[{"type": "choropleth"}]*ncols]*nrows,
    horizontal_spacing=0.005,
    vertical_spacing=0.005  # very tight
)

# Row height for placing colorbars precisely
row_height = 1 / nrows
standard_len = row_height * 0.70

# Manual colorbar placement (fixed height)
colorbar_config = {
    "Act": dict(y=1 - row_height * 0.5, len=standard_len),
    "Impl": dict(y=1 - row_height * 3.5, len=standard_len),
    "Eco": dict(y=1 - row_height * 4.5, len=standard_len),
}

used_colorbar = {"Act": False, "Impl": False, "Eco": False}

# Add maps
for idx, (map_key, label, color, dataset, zmax, dim) in enumerate(maps):
    row = idx // ncols + 1
    col = idx % ncols + 1
    geo_id = (row - 1) * ncols + col
    data = dataset[map_key]

    show_scale = False
    colorbar = None

    if not used_colorbar[dim]:
        used_colorbar[dim] = True
        cb = colorbar_config[dim]
        colorbar = dict(
            x=1.025,
            y=cb["y"],
            len=cb["len"],
            thickness=12,
            title=dict(text="USD mn", font=dict(color='black')),
            tickfont=dict(color='black')
        )
        show_scale = True

    # Define the custom colorscale based on dimension
    if dim == "Act":
        colorscale = [[0.0, 'white'], [0.3, '#8e44ad'], [1.0, '#512e5f']]
    elif dim == "Impl":
        colorscale = [[0.0, 'white'], [0.3, '#aab63d'], [1.0, '#6c7b2c']]
    elif dim == "Eco":
        colorscale = [[0.0, 'white'], [0.3, '#6bc1a4'], [1.0, '#387566']]

    fig.add_trace(
        go.Choropleth(
            locations=data["ISO3"],
            z=data["avg_annual_disbursement"],
            text=data["ISO3"],
            zmin=0,
            zmax=zmax,
            zmid= zmax*0.01,
            colorscale=colorscale,
            showscale=show_scale,
            colorbar=colorbar,
            marker_line_color='black',
            marker_line_width=0.3,
            geo=f'geo{geo_id}'
        ),
        row=row, col=col
    )

# Apply layout to each map
for i in range(1, len(maps) + 1):
    fig.update_layout(**{
        f'geo{i}': dict(
            showframe=True,
            showcoastlines=True,
            showland=True,
            landcolor='white',
            lakecolor='white',
            coastlinecolor='black',
            projection_type='equirectangular'
        )
    })

# Final figure layout
fig.update_layout(
    height=1800,
    width=1400,
    title=dict(
        text="<b>Annual Disbursement (annual average) — Key Dimensions</b>",
        x=0.5,
        font=dict(size=22, color='black')
    ),
    font=dict(color='black'),
    margin=dict(t=80, b=20, l=20, r=120),
    showlegend=False
)

for annotation in fig['layout']['annotations']:
    annotation['yshift'] = -20  # less space between title and map

fig.show(renderer="browser",config={"staticPlot": True})

fig.write_image(
    "outputs/SI_World_map_all_classes.png",
    width=1400,
    height=1800,
    scale=2
)

fig.write_image(
    "outputs/SI_World_map_all_classes.pdf",
    width=1400,
    height=1800,
    scale=2
)


# ---------------------------------------
# CONFIGURATION: COMBINED WORLD MAPS — WITH MERGED ACT CATEGORIES
# ---------------------------------------

# 1. Ensure merged ACT column exists
df['Act_SustResMgmt'] = df[['Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt', 'Act_OthMgmt']].max(axis=1)
df['Act_Protect_Resto'] = df[['Act_Protect', 'Act_Resto']].max(axis=1)

# 2. Prepare data for merged ACT categories
df_merged_act = df[df['Year'].isin(years_of_interest)].copy()
df_merged_act['USD_Disbursement'] = pd.to_numeric(df_merged_act['USD_Disbursement'], errors='coerce').fillna(0)
df_merged_act.loc[df_merged_act['USD_Disbursement'] < 0, 'USD_Disbursement'] = 0
df_merged_act['USD_Disbursement'] *= 1000  # Convert from bio to mio
df_merged_act[act_merged_cols] = df_merged_act[act_merged_cols].fillna(0).astype(int)
df_merged_act['act_count'] = df_merged_act[act_merged_cols].sum(axis=1)
df_merged_act = df_merged_act[df_merged_act['act_count'] > 0]

for col in act_merged_cols:
    df_merged_act[f'{col}_share'] = df_merged_act[col] * df_merged_act['USD_Disbursement'] / df_merged_act['act_count']

df_merged_act['ISO3'] = df_merged_act['RecipientName'].apply(get_iso3)

# 3. Aggregate ACT disbursement averages per ISO3
act_merged_results = {}
max_per_merged_act = {}
for col in act_merged_cols:
    temp = df_merged_act.groupby(['ISO3', 'Year'])[f'{col}_share'].sum().reset_index()
    avg = temp.groupby('ISO3')[f'{col}_share'].mean().reset_index()
    avg.rename(columns={f'{col}_share': 'avg_annual_disbursement'}, inplace=True)
    act_merged_results[col] = avg
    max_per_merged_act[col] = avg['avg_annual_disbursement'].max()

# 4. Compute max values for unified zmax

zmax_global = max(
    max([v for k, v in max_per_merged_act.items() if "Act_Undef" not in k]),
    max([v for k, v in max_per_impl.items() if "Impl_Undef" not in k]),
    max([v for k, v in max_per_eco.items() if "Eco_Undef" not in k])
)
zmax_act = zmax_impl = zmax_eco = zmax_global


# 5. Define all map panels (5 ACT + 3 IMPL + 4 ECO = 12 maps)
maps = [
    # Merged ACT
    ("Act_Protect_Resto", act_merged_labels_nobreak["Act_Protect_Resto"], "#8e44ad", act_merged_results, zmax_act, "Act"),
    ("Act_SustResMgmt", act_merged_labels_nobreak["Act_SustResMgmt"], "#8e44ad", act_merged_results, zmax_act, "Act"),
    ("Act_Invasiv", act_merged_labels_nobreak["Act_Invasiv"], "#8e44ad", act_merged_results, zmax_act, "Act"),
    ("Act_Pollut", act_merged_labels_nobreak["Act_Pollut"], "#8e44ad", act_merged_results, zmax_act, "Act"),

    # Implementation
    ("Impl_Regul", "Policy, regulation and governance", "#aab63d", results, zmax_impl, "Impl"),
    ("Impl_Know", "Awareness and knowledge", "#aab63d", results, zmax_impl, "Impl"),
    ("Impl_Infra", "Infrastructure", "#aab63d", results, zmax_impl, "Impl"),

    # Ecosystem
    ("Eco_CropGrass", "Cropland, rangeland, grassland, arid land", "#6bc1a4", eco_results, zmax_eco, "Eco"),
    ("Eco_Forest", "Forest", "#6bc1a4", eco_results, zmax_eco, "Eco"),
    ("Eco_SeaWater", "Marine and freshwater", "#6bc1a4", eco_results, zmax_eco, "Eco"),
    ("Eco_UrbInd", "Urban and industrial", "#6bc1a4", eco_results, zmax_eco, "Eco"),
]


# ---------------------------------------
# SI: COMBINED WORLD MAPS — WITH MERGED GOAL CLASSES
# ---------------------------------------

# 1. Calculate individual ACT shares and merged category shares
df_merged_act = df[df['Year'].isin(years_of_interest)].copy()
df_merged_act['USD_Disbursement'] = pd.to_numeric(df_merged_act['USD_Disbursement'], errors='coerce').fillna(0)
df_merged_act.loc[df_merged_act['USD_Disbursement'] < 0, 'USD_Disbursement'] = 0
df_merged_act['USD_Disbursement'] *= 1000  # Convert from bio to mio
df_merged_act[act_merged_cols] = df_merged_act[act_merged_cols].fillna(0).astype(int)
df_merged_act['act_count'] = df_merged_act[act_merged_cols].sum(axis=1)
df_merged_act = df_merged_act[df_merged_act['act_count'] > 0]

# Extend act_merged_cols to include all ACT subcategories used for merging
act_merged_cols_extended = list(set(
    act_merged_cols +
    ['Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt', 'Act_OthMgmt', 'Act_Protect', 'Act_Resto']
))

# Calculate disbursement shares for all ACT subcategories needed
for col in act_merged_cols_extended:
    df_merged_act[f'{col}_share'] = df_merged_act[col] * df_merged_act['USD_Disbursement'] / df_merged_act['act_count']


# Compute merged ACT category shares by summing individual *_share columns
sust_res_mgmt_subcats = ['Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt', 'Act_OthMgmt']
protect_resto_subcats = ['Act_Protect', 'Act_Resto']

df_merged_act['Act_SustResMgmt_share'] = df_merged_act[[f'{col}_share' for col in sust_res_mgmt_subcats]].sum(axis=1)
df_merged_act['Act_Protect_Resto_share'] = df_merged_act[[f'{col}_share' for col in protect_resto_subcats]].sum(axis=1)

df_merged_act['ISO3'] = df_merged_act['RecipientName'].apply(get_iso3)

# 3. Aggregate ACT disbursement averages per ISO3
act_merged_results = {}
max_per_merged_act = {}

# Include merged shares and individual ACTs
merged_act_share_cols = [
    'Act_SustResMgmt_share',
    'Act_Protect_Resto_share',
    'Act_Invasiv_share',
    'Act_Pollut_share',
]

for col in merged_act_share_cols:
    temp = df_merged_act.groupby(['ISO3', 'Year'])[col].sum().reset_index()
    avg = temp.groupby('ISO3')[col].mean().reset_index()
    avg.rename(columns={col: 'avg_annual_disbursement'}, inplace=True)
    act_merged_results[col.replace('_share', '')] = avg
    max_per_merged_act[col.replace('_share', '')] = avg['avg_annual_disbursement'].max()

# 4. Compute max values for unified zmax
zmax_global = max(
    max([v for k, v in max_per_merged_act.items() if "Act_Undef" not in k]),
    max([v for k, v in max_per_impl.items() if "Impl_Undef" not in k]),
    max([v for k, v in max_per_eco.items() if "Eco_Undef" not in k])
)
zmax_act = zmax_impl = zmax_eco = zmax_global

# Calculate total biodiversity investments per ISO3 (average per year, years of interest)
df_filtered = df[
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1) &
    (df['USD_Disbursement'] > 0) &
    (df['Year'].isin(years_of_interest))
].copy()

df_filtered['USD_Disbursement'] = pd.to_numeric(df_filtered['USD_Disbursement'], errors='coerce').fillna(0)
df_filtered['USD_Disbursement'] *= 1000  # Convert from bio to mio
df_filtered['ISO3'] = df_filtered['RecipientName'].apply(get_iso3)

# Group and calculate annual average per country
total_avg = (
    df_filtered.groupby(['ISO3', 'Year'])['USD_Disbursement'].sum().reset_index()
    .groupby('ISO3')['USD_Disbursement'].mean().reset_index()
)
total_avg.rename(columns={'USD_Disbursement': 'avg_annual_disbursement'}, inplace=True)

# Max for colorbar scaling
zmax_total = total_avg['avg_annual_disbursement'].max()

# 5. Define all map panels (5 ACT + 3 IMPL + 4 ECO = 12 maps)
maps = [
    # Total Biodiversity Investment (NEW)
    ("Total", f"Total (Annual average, {start_year_of_interest}_{end_year_of_interest})", "#888888", {"Total": total_avg}, zmax_total, "Total"),

    # Merged ACT
    ("Act_Protect_Resto", act_merged_labels_nobreak["Act_Protect_Resto"], "#8e44ad", act_merged_results, zmax_act, "Act"),
    ("Act_SustResMgmt", act_merged_labels_nobreak["Act_SustResMgmt"], "#8e44ad", act_merged_results, zmax_act, "Act"),
    ("Act_Invasiv", act_merged_labels_nobreak["Act_Invasiv"], "#8e44ad", act_merged_results, zmax_act, "Act"),
    ("Act_Pollut", act_merged_labels_nobreak["Act_Pollut"], "#8e44ad", act_merged_results, zmax_act, "Act"),

    # Implementation
    ("Impl_Regul", "Policy, regulation and governance", "#aab63d", results, zmax_impl, "Impl"),
    ("Impl_Know", "Awareness and knowledge", "#aab63d", results, zmax_impl, "Impl"),
    ("Impl_Infra", "Infrastructure", "#aab63d", results, zmax_impl, "Impl"),

    # Ecosystem
    ("Eco_CropGrass", "Cropland, rangeland, grassland, arid land", "#6bc1a4", eco_results, zmax_eco, "Eco"),
    ("Eco_Forest", "Forest", "#6bc1a4", eco_results, zmax_eco, "Eco"),
    ("Eco_SeaWater", "Marine and freshwater", "#6bc1a4", eco_results, zmax_eco, "Eco"),
    ("Eco_UrbInd", "Urban and industrial", "#6bc1a4", eco_results, zmax_eco, "Eco"),
]

# 6. Define layout grid
ncols = 3
nrows = (len(maps) + ncols - 1) // ncols

fig = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=[f"<b>{label}</b>" for _, label, *_ in maps],
    specs=[[{"type": "choropleth"}] * ncols] * nrows,
    horizontal_spacing=0.005,
    vertical_spacing=0.005
)

row_height = 1 / nrows
standard_len = row_height * 0.70


# Align all colorbars vertically in bottom-right corner
colorbar_config = {
    "Total": dict(x=0.86, y=-0.12, len=0.15, yanchor="bottom"),  # Leftmost
    "Act":   dict(x=0.9,  y=-0.12, len=0.15, yanchor="bottom"),
    "Impl":  dict(x=0.94, y=-0.12, len=0.15, yanchor="bottom"),
    "Eco":   dict(x=0.98, y=-0.12, len=0.15, yanchor="bottom"),
}

used_colorbar = {"Total": False, "Act": False, "Impl": False, "Eco": False}

# 7. Add maps to figure
for idx, (map_key, label, color, dataset, zmax, dim) in enumerate(maps):
    row = idx // ncols + 1
    col = idx % ncols + 1
    geo_id = (row - 1) * ncols + col
    data = dataset[map_key]

    show_scale = False
    colorbar = None

    if not used_colorbar[dim]:
        used_colorbar[dim] = True
        cb = colorbar_config[dim]

        if dim == "Eco":
            colorbar = dict(
                x=cb["x"],
                y=cb["y"],
                len=cb["len"],
                yanchor=cb["yanchor"],
                thickness=12,
                title=dict(text="_", font=dict(color='white')),
                tickfont=dict(color='black', size=12)
            )

        elif dim == "Total":
            colorbar = dict(
                x=cb["x"],
                y=cb["y"],
                len=cb["len"],
                yanchor=cb["yanchor"],
                thickness=12,
                tickvals=[],
                ticktext=[],
                title=dict(text="Annual average (USD bn)", font=dict(color='black')),
                tickfont=dict(color='white', size=12)
            )

        else:
            colorbar = dict(
                x=cb["x"],
                y=cb["y"],
                len=cb["len"],
                yanchor=cb["yanchor"],
                thickness=12,
                tickvals=[],
                ticktext=[],
                title=dict(text="_", font=dict(color='white')),
                tickfont=dict(color='white', size=12)
            )

        show_scale = True

    # Custom colorscale by dimension
    if dim == "Total":
        colorscale = [[0.0, 'white'], [0.2, '#b0b0b0'], [1.0, '#404040']]
    elif dim == "Act":
        colorscale = [[0.0, 'white'], [0.2, '#8e44ad'], [1.0, '#512e5f']]
    elif dim == "Impl":
        colorscale = [[0.0, 'white'], [0.2, '#aab63d'], [1.0, '#6c7b2c']]
    elif dim == "Eco":
        colorscale = [[0.0, 'white'], [0.2, '#6bc1a4'], [1.0, '#387566']]

    fig.add_trace(
        go.Choropleth(
            locations=data["ISO3"],
            z=data["avg_annual_disbursement"]/1000,
            text=data["ISO3"],
            zmin=0,
            zmax=zmax/1000,
            zmid=zmax * 0.01,
            colorscale=colorscale,
            showscale=show_scale,
            colorbar=colorbar,
            marker_line_color='black',
            marker_line_width=0.3,
            geo=f'geo{geo_id}'
        ),
        row=row, col=col
    )

# 8. Apply layout for each map
for i in range(1, len(maps) + 1):
    fig.update_layout(**{
        f'geo{i}': dict(
            showframe=True,
            showcoastlines=True,
            showland=True,
            landcolor='white',
            lakecolor='white',
            coastlinecolor='black',
            projection_type='equirectangular',
        lataxis = dict(range=[-70, 90])  # Cut off Antarctica
        )
    })

# 9. Final figure layout
fig.update_layout(
    height=300 * nrows + 200,
    width=1400,
    margin=dict(t=40, b=60, l=20, r=120),
    showlegend=False
)

# Adjust subtitle spacing
for annotation in fig['layout']['annotations']:
    annotation['yshift'] = -5
    annotation['font'] = dict(size=18, color='black')

# 10. Show and export
fig.show(renderer="browser",config={"staticPlot": True})

fig.write_image("outputs/SI_World_map_merged_goal.png", width=1400, height=300 * nrows + 200, scale=2)
fig.write_image("outputs/SI_World_map_merged_goal.pdf", width=1400, height=300 * nrows + 200, scale=2)


# ---------------------------------------
# CONFIGURATION INDIVIDUAL WORLD MAPS – MERGED ACT CATEGORIES
# ---------------------------------------

# Loop through each entry in 'maps' list and export individually
for idx, (map_key, label, color, dataset, zmax, dim) in enumerate(maps):
    data = dataset[map_key]

    # Choose colorscale per dimension
    if dim == "Total":
        colorscale = [[0.0, 'white'], [0.2, '#b0b0b0'], [1.0, '#404040']]
    elif dim == "Act":
        colorscale = [[0.0, 'white'], [0.2, '#8e44ad'], [1.0, '#512e5f']]
    elif dim == "Impl":
        colorscale = [[0.0, 'white'], [0.2, '#aab63d'], [1.0, '#6c7b2c']]
    elif dim == "Eco":
        colorscale = [[0.0, 'white'], [0.2, '#6bc1a4'], [1.0, '#387566']]

    # Create a single-map figure
    fig_single = go.Figure()

    fig_single.add_trace(
        go.Choropleth(
            locations=data["ISO3"],
            z=data["avg_annual_disbursement"],
            text=data["ISO3"],
            zmin=0,
            zmax=zmax,
            zmid=zmax * 0.01,
            colorscale=colorscale,
            showscale=False,
            marker_line_color='black',
            marker_line_width=0.3
        )
    )

    # Title in layout to act as the subtitle
    fig_single.update_layout(
        title=dict(text=f"<b>{label}</b>", x=0.5, xanchor="center", font=dict(size=36)),
        geo=dict(
            showframe=True,
            showcoastlines=True,
            showland=True,
            landcolor='white',
            lakecolor='white',
            coastlinecolor='black',
            projection_type='equirectangular',
            lataxis=dict(range=[-70, 90])  # cut off Antarctica
        ),
        margin=dict(t=60, b=20, l=0, r=0),
        height=600,
        width=1000
    )


# ---------------------------------------
# SI WORLD MAP – MERGED ACT CATEGORIES
# ---------------------------------------

# 1. Ensure Act_SustResMgmt exists
df['Act_SustResMgmt'] = df[['Act_Agri', 'Act_ForestMgmt', 'Act_Fish', 'Act_WaterMgmt', 'Act_OthMgmt']].max(axis=1)
df['Act_Protect_Resto'] = df[['Act_Protect', 'Act_Resto']].max(axis=1)

# 2. Prepare filtered DataFrame
df_merged_act = df[df['Year'].isin(years_of_interest)].copy()
df_merged_act['USD_Disbursement'] = pd.to_numeric(df_merged_act['USD_Disbursement'], errors='coerce').fillna(0)
df_merged_act.loc[df_merged_act['USD_Disbursement'] < 0, 'USD_Disbursement'] = 0
df_merged_act['USD_Disbursement'] *= 1000  # From bio to mio

# Ensure merged ACT columns are integers
df_merged_act[act_merged_cols] = df_merged_act[act_merged_cols].fillna(0).astype(int)

# Keep rows with at least one ACT label
df_merged_act['act_count'] = df_merged_act[act_merged_cols].sum(axis=1)
df_merged_act = df_merged_act[df_merged_act['act_count'] > 0]

# Split disbursements proportionally
for col in act_merged_cols:
    df_merged_act[f'{col}_share'] = df_merged_act[col] * df_merged_act['USD_Disbursement'] / df_merged_act['act_count']

# ISO3 conversion
df_merged_act['ISO3'] = df_merged_act['RecipientName'].apply(get_iso3)

# 3. Aggregate to compute average annual disbursement per ISO3
act_merged_results = {}
max_per_merged_act = {}

for col in act_merged_cols:
    temp = df_merged_act.groupby(['ISO3', 'Year'])[f'{col}_share'].sum().reset_index()
    avg = temp.groupby('ISO3')[f'{col}_share'].mean().reset_index()
    avg.rename(columns={f'{col}_share': 'avg_annual_disbursement'}, inplace=True)
    act_merged_results[col] = avg
    max_per_merged_act[col] = avg['avg_annual_disbursement'].max()

# 4. Plotting setup
num_merged = len(act_merged_order)
rows_merged = (num_merged + 1) // 2
fig_merged = make_subplots(
    rows=rows_merged, cols=2,
    subplot_titles=[f"<b>{act_merged_labels_nobreak[col]}</b>" for col in act_merged_order],
    specs=[[{"type": "choropleth"}, {"type": "choropleth"}]] * rows_merged,
    horizontal_spacing=0.15,
    vertical_spacing=0.08
)

merged_positions = [(r, c) for r in range(1, rows_merged + 1) for c in range(1, 3)]
colorbars = [get_colorbar_settings(c, r, rows_merged) for (r, c) in merged_positions]

# 5. Add merged ACT maps
for (col_name, color), (row, col_pos), cbar in zip(zip(act_merged_order, act_merged_colors), merged_positions, colorbars):
    data = act_merged_results[col_name]
    zmax = max_per_merged_act[col_name]

    fig_merged.add_trace(
        go.Choropleth(
            locations=data["ISO3"],
            z=data["avg_annual_disbursement"],
            text=data["ISO3"],
            colorscale=[[0.0, 'white'], [1.0, color]],
            zmin=0,
            zmax=zmax,
            colorbar=cbar,
            showscale=True,
            geo=f'geo{(row - 1) * 2 + col_pos}'
        ),
        row=row, col=col_pos
    )

# 6. Layout updates
fig_merged.update_layout(
    height=450 * rows_merged,
    width=1200,
    title=dict(
        text=f"<b>Annual disbursement by goal ({start_year_of_interest}-{end_year_of_interest} average, Merged Categories)</b>",
        x=0.5,
        font=dict(size=20, color='black')
    ),
    margin=dict(t=90, b=30, l=20, r=30),
    showlegend=False,
    font=dict(color='black')
)

fig_merged.update_layout(**{f'geo{i}': standard_geo_layout() for i in range(1, len(act_merged_order) + 1)})

# Adjust spacing between titles and maps
for annotation in fig_merged['layout']['annotations']:
    annotation['yshift'] = -20

# Show the figure
fig_merged.show(renderer="browser",config={'staticPlot': True})

# Save PNG
fig_merged.write_image(
    "outputs/SI_World_map_goal_classes_invasive species.png",
    width=1200,
    height=450 * rows_merged,
    scale=2
)

# Save PDF
fig_merged.write_image(
    "outputs/SI_World_map_goal_classes_invasive species.pdf",
    width=1200,
    height=450 * rows_merged,
    scale=2
)

# ---------------------------------------
# TABLE Yearly Summary Statistics Table 2000-2023
# ---------------------------------------

summary = pd.DataFrame(index=range(2000, 2024))

# Combine all
summary = summary.join(all_stats, how='left')
summary = summary.join(biodiv_stats, how='left')
summary = summary.join(bert_stats, how='left')

# Convert disbursements from Bio to '000 USD (thousands)
cols_to_scale = [col for col in summary.columns if 'Disb' in col]
summary[cols_to_scale] = summary[cols_to_scale] * 1000

# Round to two decimals
summary = summary.round(3)

# Transpose table for better orientation
summary_t = summary.T


# ---------------------------------------
# TABLE: Boxplot: BiodivBERT-LLM Disbursements by Year (2000–2023)
#------------------------------------------

# Copy all BiodivBERT-LLM projects, but ONLY those with USD_Disbursement > 0  ← NEW/CHANGED
bert_llm_df = df[
    (df['binary_label_biodiversity_impact'] == 1)
    & (df['No_Biodiv'] != 1)
    & (df['USD_Disbursement'] > 0)
].copy()

biodivbert_box_df = bert_llm_df.copy()
biodivbert_box_df['USD_Disbursement_Mio'] = biodivbert_box_df['USD_Disbursement'] * 1000
filtered_box_df = biodivbert_box_df[biodivbert_box_df['Year'].between(2000, 2023)]

# --- 1) Compute & export yearly stats (matches plot filter) ---
clean = filtered_box_df.loc[
    filtered_box_df["USD_Disbursement_Mio"].notna() &
    (filtered_box_df["USD_Disbursement_Mio"] > 0),
    ["Year", "USD_Disbursement_Mio"]
].copy()

# make sure Year is numeric int
clean["Year"] = pd.to_numeric(clean["Year"], errors="coerce")
clean = clean.dropna(subset=["Year"])
clean["Year"] = clean["Year"].astype(int)

# year range (already defined above)
years = list(range(2000, 2024))

stats = (
    clean.groupby("Year")["USD_Disbursement_Mio"]
         .agg(
             count="count",
             min="min",
             p10=lambda s: s.quantile(0.10),
             q1=lambda s: s.quantile(0.25),
             median="median",
             mean="mean",
             q3=lambda s: s.quantile(0.75),
             p90=lambda s: s.quantile(0.90),
             max="max",
             iqr=lambda s: s.quantile(0.75) - s.quantile(0.25),
             std="std",
         )
         .reindex(years)
         .round(6)
)

# write to Excel
os.makedirs(
    "./outputs", exist_ok=True)
out_path = "outputs/Bidiv_Cube_yearly_stats.xlsx"
with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    stats.to_excel(writer, sheet_name="stats", merge_cells=False)

print(f"SI Table: Exported Biodiv Cube yearly stats to {out_path}")

#------------------------------------------
# CONFIGURATION
#------------------------------------------

# Create proportionally distributed disbursement and count data
act_disb = calc_group_split(df, act_cols, 'USD_Disbursement')
impl_disb = calc_group_split(df, impl_cols, 'USD_Disbursement')
eco_disb = calc_group_split(df, eco_cols, 'USD_Disbursement')

# Filter before counting
df_nonzero = df[df['USD_Disbursement'] > 0]

# Then compute counts
act_count = calc_group_split(df_nonzero, act_cols, 'Count')
impl_count = calc_group_split(df_nonzero, impl_cols, 'Count')
eco_count = calc_group_split(df_nonzero, eco_cols, 'Count')

# Biodiversity lines too
bio_act_count_line = calc_group_split(df_nonzero, act_cols, 'Count', filter_biodiv=True).sum(axis=1)
bio_impl_count_line = calc_group_split(df_nonzero, impl_cols, 'Count', filter_biodiv=True).sum(axis=1)
bio_eco_count_line = calc_group_split(df_nonzero, eco_cols, 'Count', filter_biodiv=True).sum(axis=1)

# Biodiversity lines (also proportional)
bio_act_disb_line = calc_group_split(df, act_cols, 'USD_Disbursement', filter_biodiv=True).sum(axis=1)
bio_impl_disb_line = calc_group_split(df, impl_cols, 'USD_Disbursement', filter_biodiv=True).sum(axis=1)
bio_eco_disb_line = calc_group_split(df, eco_cols, 'USD_Disbursement', filter_biodiv=True).sum(axis=1)

# Plotting Disbursement Bar Charts
ffig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

for ax, data, bio_line, title, colors, col_order, labels in zip(
        axs,
        [act_disb, impl_disb, eco_disb],
        [bio_act_disb_line, bio_impl_disb_line, bio_eco_disb_line],
        ['Goal ', 'Instrument', 'Ecosystem'],
        [act_colors, impl_colors, eco_colors],
        [act_order, impl_order, eco_order],
        [act_labels_nobreak, impl_labels_nobreak, eco_labels_nobreak]
):
    valid_cols = [col for col in col_order if col in data.columns]
    data = data.loc[:, valid_cols]
    renamed_cols = [labels.get(col, col) for col in valid_cols]
    data.columns = renamed_cols

    color_map = [colors[col_order.index(col)] for col in valid_cols]

    # --- stacked AREA chart using explicit x (years) ---
    years = pd.to_numeric(data.index, errors="coerce")
    order = np.argsort(years)
    years = years[order]
    data = data.iloc[order]

    ax.stackplot(
        years,
        [data[col].values for col in data.columns],
        colors=color_map,
        labels=data.columns
    )

    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_ylabel('Disbursements [USD bn]')
    ax.set_xlabel(' ')
    ax.set_ylim(0, 20)
    ax.legend(loc='upper left', fontsize=10)
    ax.margins(x=0.02, y=0)

    # Show only every 5th year
    years_int = years.astype(int)
    xtick_years = sorted({y for y in years_int if y % 5 == 0})
    ax.set_xticks(xtick_years)
    ax.set_xticklabels([str(y) for y in xtick_years], rotation=0)

plt.rcParams.update({'font.size': 10})
plt.tight_layout()
output_path= "outputs/SI_disbursement_detailed_goal_classes_2000-2023.png"
plt.savefig(output_path, dpi=300)
plt.savefig(output_path, dpi=300)
plt.show()
print(f"SI Figure: Saved disbursement area charts to {output_path}")

#------------------------------------------
# CONFIGURATION
#------------------------------------------

def prep_disb_table(data, col_order, labels, years=None):

    valid_cols = [c for c in col_order if c in data.columns]
    tbl = data.loc[:, valid_cols].copy()
    tbl.columns = [labels.get(c, c) for c in valid_cols]

    # --- FIX: coerce the index to numeric years with a mask ---
    yr = pd.to_numeric(pd.Index(tbl.index).astype(str), errors='coerce')
    mask = ~pd.isna(yr)
    tbl = tbl.iloc[mask].copy()
    tbl.index = yr[mask].astype(int)
    tbl = tbl.sort_index()

    if years is not None:
        tbl = tbl.reindex(years)

    tbl["Total"] = tbl.sum(axis=1, min_count=1)

    return tbl.round(3)

years = list(range(2000, 2024))

goal_tbl = prep_disb_table(act_disb,  act_order,  act_labels_nobreak,  years)
instr_tbl = prep_disb_table(impl_disb, impl_order, impl_labels_nobreak, years)
eco_tbl   = prep_disb_table(eco_disb,  eco_order,  eco_labels_nobreak,  years)

# ---------- write to Excel ---------
os.makedirs(
    "./outputs", exist_ok=True)
out_path = "outputs/SI_disbursement_table.xlsx"

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    # Wide tables (years as index, categories as columns + Total)
    goal_tbl.to_excel(writer, sheet_name="Goal (wide)")
    instr_tbl.to_excel(writer, sheet_name="Instrument (wide)")
    eco_tbl.to_excel(writer, sheet_name="Ecosystem (wide)")

    # Long tables (one value per row)
    (goal_tbl.reset_index(names="Year")
             .melt(id_vars="Year", var_name="Category", value_name="USD bn")
             .to_excel(writer, sheet_name="Goal (long)", index=False))
    (instr_tbl.reset_index(names="Year")
              .melt(id_vars="Year", var_name="Category", value_name="USD bn")
              .to_excel(writer, sheet_name="Instrument (long)", index=False))
    (eco_tbl.reset_index(names="Year")
            .melt(id_vars="Year", var_name="Category", value_name="USD bn")
            .to_excel(writer, sheet_name="Ecosystem (long)", index=False))

print(f"SI Table: Exported yearly disbursements to {out_path}")


# ------------------------------------------------------------
# MAIN: INDIVIDUAL WORLD MAPS
# ------------------------------------------------------------

EXCEL_PATH = "outputs/avg_annual_disbursement_by_country_and_subclass.xlsx"

# Output folder
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COLUMN CONFIGURATION
sheet_configs = [
    {
        "sheet": "Act",
        "dim": "Act",
        "columns": {
            "%_Invasive species": "Invasive species management",
            "%_Pollution control": "Pollution control",
            "%_Protection, conservation & restoration": "Protection, conservation & restoration",
            "%_Sustainable Management": "Sustainable management",
        },
    },
    {
        "sheet": "Impl",
        "dim": "Impl",
        "columns": {
            "%_Policy, regulatory and governance": "Policy, regulation and governance",
            "%_Awareness and knowledge": "Awareness and knowledge",
            "%_Infrastructure building": "Infrastructure",
        },
    },
    {
        "sheet": "Eco",
        "dim": "Eco",
        "columns": {
            "%_Crop-, range-, grass-, and arid land": "Cropland, rangeland, grassland, arid land",
            "%_Forest": "Forest",
            "%_Marine and freshwater": "Marine and freshwater",
            "%_Urban and industrial": "Urban and industrial",
            "%_Undefined": "Undefined ecosystem type",
        },
    },
]

# HELPER: COLORSCALE PER DIMENSION
def get_colorscale_for_dim(dim: str):
    if dim == "Act":
        return [[0.0, 'white'], [0.6, '#8e44ad'], [1.0, '#512e5f']]
    elif dim == "Impl":
        return [[0.0, 'white'], [0.6, '#aab63d'], [1.0, '#6c7b2c']]
    elif dim == "Eco":
        return [[0.0, 'white'], [0.6, '#6bc1a4'], [1.0, '#387566']]
    else:
        # Fallback grayscale
        return [[0.0, 'white'], [0.6, '#b0b0b0'], [1.0, '#404040']]


# MAIN LOOP: CREATE INDIVIDUAL PERCENTAGE MAPS
for config in sheet_configs:
    sheet_name = config["sheet"]
    dim = config["dim"]
    column_mapping = config["columns"]

    df_sheet = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)

    if "ISO3" not in df_sheet.columns:
        raise ValueError(f'Sheet "{sheet_name}" must contain an "ISO3" column.')

    for col_name, label in column_mapping.items():
        if col_name not in df_sheet.columns:
            print(f'Warning: column "{col_name}" not found in sheet "{sheet_name}". Skipping.')
            continue

        data = df_sheet[["ISO3", col_name]].copy()
        data = data.dropna(subset=[col_name])

        data[col_name] = pd.to_numeric(data[col_name], errors="coerce")
        data = data.dropna(subset=[col_name])

        if data.empty:
            print(f'No valid data for column "{col_name}" in sheet "{sheet_name}". Skipping.')
            continue

        # Percentages are set as fractions between 0 and 1
        z_values = data[col_name].values
        zmin = 0.0
        zmax = 1.0  # full 0–1 range (0–100%) for comparability across maps
        zmid = 0.5  # mid-point for color stretching

        colorscale = get_colorscale_for_dim(dim)

        # Create figure
        fig_single = go.Figure()

        fig_single.add_trace(
            go.Choropleth(
                locations=data["ISO3"],
                z=z_values,
                text=data["ISO3"],
                zmin=zmin,
                zmax=zmax,
                zmid=zmid,
                colorscale=colorscale,
                showscale=False,  # no legend/colorbar on the individual maps
                marker_line_color='black',
                marker_line_width=0.3,
            )
        )

        # Layout similar to  existing single-map style
        fig_single.update_layout(
            title=dict(
                text=f"<b>{label}</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=36),
            ),
            geo=dict(
                showframe=True,
                showcoastlines=True,
                showland=True,
                landcolor='white',
                lakecolor='white',
                coastlinecolor='black',
                projection_type='equirectangular',
                lataxis=dict(range=[-70, 90]),  # cut off Antarctica
            ),
            margin=dict(t=60, b=20, l=0, r=0),
            height=600,
            width=1000,
        )

        # Safe filename: remove problematic characters
        safe_col_key = (
            col_name.replace("%", "pct")
            .replace(" ", "_")
            .replace(",", "")
            .replace("-", "")
            .replace("&", "and")
            .replace("__", "_")
        )

        filename = os.path.join(
            OUTPUT_DIR,
            f"MAIN_world_map_pct_{sheet_name}_{safe_col_key}.png"
        )

        # Export
        fig_single.write_image(filename, scale=2)
        print(f'Main Figure: Exported World Maps for main paper to: {filename}')


# COLORBAR LEGEND – Clean version (no titles, no % labels, tight spacing)
legend_fig = go.Figure()

# No tick values and no tick labels
tickvals = []
ticktext = []

# X positions for the colorbars
x_positions = [0.40, 0.50, 0.60]

dimensions = ["Act", "Impl", "Eco"]

for x, dim in zip(x_positions, dimensions):
    legend_fig.add_trace(
        go.Heatmap(
            z=[[0, 1]],
            showscale=True,
            colorscale=get_colorscale_for_dim(dim),
            opacity=0,
            colorbar=dict(
                x=x,
                y=0.5,
                len=0.8,
                thickness=18,
                tickvals=tickvals,
                ticktext=ticktext,
                outlinewidth=0,
            ),
            hoverinfo="skip",
        )
    )

legend_fig.update_layout(
    title=dict(
        text="<b>Color scales – Share of country biodiversity funding</b>",
        x=0.5,
        xanchor="center",
        font=dict(size=22),
    ),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    margin=dict(t=70, b=20, l=20, r=20),
    height=350,
    width=700,
    paper_bgcolor="white",
    plot_bgcolor="white",
)

legend_path = os.path.join(OUTPUT_DIR, "colorbar_legend_percentages.png")
legend_fig.write_image(legend_path, scale=2)

# ------------------------------------------------
# TABLE: TOP DONORS (2000–2024) FOR BIODIVERSITY SUBSET, One XLSX sheet, four titled blocks
# ------------------------------------------------

YEAR_MIN_AGG, YEAR_MAX_AGG = 2000, 2024

df_pos_top = df[
    (df['USD_Disbursement'] > 0) &
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1) &
    (df['Year'].between(YEAR_MIN_AGG, YEAR_MAX_AGG, inclusive="both"))
].copy()

# Make sure Bi_Multi is numeric
df_pos_top["Bi_Multi"] = pd.to_numeric(df_pos_top["Bi_Multi"], errors="coerce")

# --- Map codes to the merged donor labels (same logic as above) ---
donor_fold_map = {
    1: "Bilateral", 7: "Bilateral",
    4: "Multilateral",
    6: "Private",
    3: "Other", 5: "Other", 8: "Other"
}
order_donor_labels = ["Bilateral", "Multilateral", "Private", "Other"]

df_pos_top["DonorLabel"] = df_pos_top["Bi_Multi"].map(donor_fold_map)

# Drop rows without a mapped label or missing DonorName
df_pos_top = df_pos_top[df_pos_top["DonorLabel"].notna() & df_pos_top["DonorName"].notna()].copy()

# --- Helper: compute top N donors by label ---
def top_donors_by_label(df_in, label, topn=20):
    sub = df_in[df_in["DonorLabel"] == label]
    if sub.empty:
        return pd.DataFrame(columns=["Rank", "DonorName", "Total_USD_Disbursement"])
    agg = (sub.groupby("DonorName", dropna=False)["USD_Disbursement"]
              .sum()
              .sort_values(ascending=False)
              .head(topn)
              .reset_index())
    agg.rename(columns={"USD_Disbursement": "Total_USD_Disbursement"}, inplace=True)
    agg.insert(0, "Rank", range(1, len(agg) + 1))
    return agg

# Build the four tables
top_bilateral     = top_donors_by_label(df_pos_top, "Bilateral", topn=20)
top_multilateral  = top_donors_by_label(df_pos_top, "Multilateral", topn=20)
top_private       = top_donors_by_label(df_pos_top, "Private", topn=20)
top_other         = top_donors_by_label(df_pos_top, "Other", topn=20)

# Optional: ensure numeric column is clearly numeric and round if desired
for t in (top_bilateral, top_multilateral, top_private, top_other):
    if not t.empty:
        t["Total_USD_Disbursement"] = pd.to_numeric(t["Total_USD_Disbursement"], errors="coerce")

# --- Write a single-sheet Excel with four titled blocks ---
os.makedirs(
    "./outputs", exist_ok=True)
out_xlsx_top = "outputs/FINAL_TOP_DONORS_2000_2024.xlsx"

with pd.ExcelWriter(out_xlsx_top, engine="xlsxwriter") as writer:
    sheet_name = "TopDonors"
    row_cursor = 0
    col_cursor = 0

    workbook  = writer.book
    title_fmt = workbook.add_format({"bold": True, "font_size": 12})

    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    def write_block(ws, wr, start_row, start_col, title, df_block):
        ws.write(start_row, start_col, title, title_fmt)
        start_row += 1
        if df_block is not None and not df_block.empty:
            df_block.to_excel(wr, sheet_name=sheet_name, startrow=start_row, startcol=start_col, index=False)
            start_row += df_block.shape[0] + 3
        else:
            ws.write(start_row, start_col, "(no data)")
            start_row += 3
        return start_row

    header = f"Top 20 donors by cumulated USD_Disbursement, {YEAR_MIN_AGG}–{YEAR_MAX_AGG}, biodiversity subset (USD_Disbursement>0, binary_label=1 & No_Biodiv!=1); donor classes per Bi_Multi fold map."
    worksheet.write(row_cursor, col_cursor, header, title_fmt)
    row_cursor += 2

    row_cursor = write_block(worksheet, writer, row_cursor, col_cursor,
                             "Bilateral — Top 20", top_bilateral)

    row_cursor = write_block(worksheet, writer, row_cursor, col_cursor,
                             "Multilateral — Top 20", top_multilateral)

    row_cursor = write_block(worksheet, writer, row_cursor, col_cursor,
                             "Private — Top 20", top_private)

    row_cursor = write_block(worksheet, writer, row_cursor, col_cursor,
                             "Other — Top 20", top_other)

print(f"SI Table: Exported top 20 donor table: {out_xlsx_top}")

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

YEARS = range(2000, 2024)

# 1) Filter base data once
df_nonzero = df.loc[df['USD_Disbursement'].notna() & (df['USD_Disbursement'] > 0)].copy()

# 2) Recompute all aggregates from filtered data
act_count_f  = calc_group_split(df_nonzero, act_cols,  'Count')
impl_count_f = calc_group_split(df_nonzero, impl_cols, 'Count')
eco_count_f  = calc_group_split(df_nonzero, eco_cols,  'Count')

# Biodiversity counts (unmerged) — these drive stacked areas only, not the Rio line
bio_act_count_f  = calc_group_split(df_nonzero, act_cols,  'Count', filter_biodiv=True)
bio_impl_count_f = calc_group_split(df_nonzero, impl_cols, 'Count', filter_biodiv=True)
bio_eco_count_f  = calc_group_split(df_nonzero, eco_cols,  'Count', filter_biodiv=True)

# Lines = totals per time index (sum across columns) — kept for export/diagnostics
bio_act_count_line_f  = bio_act_count_f.sum(axis=1)
bio_impl_count_line_f = bio_impl_count_f.sum(axis=1)
bio_eco_count_line_f  = bio_eco_count_f.sum(axis=1)

# Merged ACT counts + merged ACT biodiv line (for stacked areas)
act_merged_count_f = merge_act_groups(act_count_f, kind='count')
bio_act_merged_count_line_f = merge_act_groups(bio_act_count_f).sum(axis=1)


# 2b) SINGLE SOURCE OF TRUTH for the Rio Marker line (COUNTS)
#     Filter: USD_Disbursement > 0 and Biodiversity in {1,2}
rio_count_year = (
    df_nonzero.loc[df_nonzero['Biodiversity'].isin([1, 2])]
       .groupby('Year')
       .size()
       .reindex(YEARS, fill_value=0)
)



# CONFIGURATION
# ------------------------------------------------

# 4) Export UNMERGED to Excel (+ Rio line for reference)
with pd.ExcelWriter(
        "outputs/SI_project_count_table.xlsx", engine='xlsxwriter') as writer:
    act_count_f.to_excel(writer,  sheet_name='Goal_Project_Count')
    impl_count_f.to_excel(writer, sheet_name='Instrument_Project_Count')
    eco_count_f.to_excel(writer,  sheet_name='Ecosystem_Project_Count')

    bio_act_count_line_f.to_frame(name='BiodivBERT_Line').to_excel(writer,  sheet_name='Bio_Goal_Line')
    bio_impl_count_line_f.to_frame(name='BiodivBERT_Line').to_excel(writer, sheet_name='Bio_Instrument_Line')
    bio_eco_count_line_f.to_frame(name='BiodivBERT_Line').to_excel(writer,  sheet_name='Bio_Ecosystem_Line')

    rio_count_year.to_frame(name='Rio_Marker_Count').to_excel(writer, sheet_name='Rio_Marker_Line')

# ____________________________________________________________________
# SI: Project_count_detailed_goal
# ____________________________________________________________________


# 6) Compute “All Biodiversity Projects” (BiodivBERT) for first panel
df_biodiv_all = df_nonzero[
    (df_nonzero['binary_label_biodiversity_impact'] == 1) &
    (df_nonzero['No_Biodiv'] != 1)
].copy()

# Count total number of projects per year (USD_Disbursement > 0 and BiodivBERT=1, No_Biodiv!=1)
bio_all_per_year = (
    df_biodiv_all.groupby("Year")["Count"].sum()
      .reindex(YEARS)
      .fillna(0)
)

# 7) Project Count Area Charts (UNMERGED) + “All Biodiv” panel
#     — overlay SAME Rio line in panel 1 and panels 2–4
fig, axs = plt.subplots(1, 4, figsize=(22, 6), sharex=True)

# ---- FIRST PANEL: All Biodiversity Projects (BiodivBERT area) + SAME Rio line
ax = axs[0]
years = pd.Index(YEARS)
ax.fill_between(years, bio_all_per_year.reindex(years).fillna(0), color='lightgrey', alpha=0.9,
                label='ODF-BiodivBERT')
ax.plot(
    years,
    rio_count_year.reindex(years).fillna(0),
    linestyle='--', color='red', linewidth=3,
    label='Biodiversity Rio Marker (1&2)'
)

ax.margins(x=0.02, y=0)
ax.set_title("ODF-BiodivBERT", fontsize=18, fontweight='bold', pad=15)
ax.set_ylabel("Project Count")
ax.set_xlabel(" ")
ax.legend(loc='upper left', fontsize=10)

# ---- NEXT 3 PANELS (Goal / Instrument / Ecosystem) — SAME Rio line
for ax, data, title, colors, col_order, labels_dict in zip(
        axs[1:],  # skip first panel
        [act_count_f, impl_count_f, eco_count_f],
        ['Goal', 'Instrument', 'Ecosystem'],
        [act_colors, impl_colors, eco_colors],
        [act_order, impl_order, eco_order],
        [act_labels_nobreak, impl_labels_nobreak, eco_labels_nobreak]
):
    valid_cols = [col for col in col_order if col in data.columns]
    data = data.loc[:, valid_cols]
    color_map = [colors[col_order.index(col)] for col in valid_cols]
    renamed_cols = [labels_dict.get(col, col) for col in valid_cols]
    data.columns = renamed_cols
    data.plot.area(ax=ax, stacked=True, color=color_map)

    ax.margins(x=0.02, y=0)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_ylabel("Project Count")
    ax.set_xlabel(" ")
    ax.legend(loc='upper left', fontsize=10)

plt.rcParams.update({'font.size': 10})
plt.tight_layout()

output_path = "./outputs/SI_Project_count_detailed_goal.png"
plt.savefig(output_path, dpi=300)
plt.savefig("outputs/SI_Project_count_detailed_goal.pdf", dpi=300)

plt.show()

print(f"SI Figure: Saved project count area charts to {output_path} and .pdf")

# ---------------------------
# SHARED CONFIGURATION
# ---------------------------
os.makedirs(
    "./outputs", exist_ok=True)

YEARS = list(range(2000, 2024))
years_index = pd.Index(YEARS, dtype=int)
years = years_index

# Ticks every 5 years
xtick_positions = [i for i, y in enumerate(YEARS) if y % 5 == 0]
xtick_labels    = [str(YEARS[i]) for i in xtick_positions]

TITLE_FONTSIZE  = 18
LABEL_FONTSIZE  = 12
LEGEND_FONTSIZE = 12
BASE_FONT       = 12
plt.rcParams.update({'font.size': BASE_FONT})

# Disbursement y-limit (match the bar charts)
DISB_YMAX = 22

# Column for Rio Marker
RIO_MARKER_COL = "Biodiversity"

def _prep_stacked(df_in, col_order, colors, labels_dict):
    """Return (df_plot, color_list) with columns/order/labels resolved and reindexed to YEARS."""
    valid_cols = [c for c in col_order if c in df_in.columns]
    data = df_in.loc[:, valid_cols].copy()
    # rename columns for legend
    renamed = [labels_dict.get(c, c) for c in valid_cols]
    data.columns = renamed
    color_list = [colors[col_order.index(c)] for c in valid_cols]
    data = data.reindex(YEARS).fillna(0)
    return data, color_list

# 1) PROJECT COUNT DATA
# Base: only projects with positive disbursement
df_nonzero = df.loc[df['USD_Disbursement'].notna() & (df['USD_Disbursement'] > 0)].copy()

# Unmerged counts
act_count_f  = calc_group_split(df_nonzero, act_cols,  'Count')
impl_count_f = calc_group_split(df_nonzero, impl_cols, 'Count')
eco_count_f  = calc_group_split(df_nonzero, eco_cols,  'Count')

# Merged ACT counts for panel 2
act_merged_count_f = merge_act_groups(act_count_f, kind='count')

# Unified Rio COUNT line
rio_count_year = (
    df_nonzero.loc[df_nonzero[RIO_MARKER_COL].isin([1, 2])]
      .groupby('Year')
      .size()
      .reindex(YEARS, fill_value=0)
)

# “All Biodiversity Projects” (BiodivBERT) for panel 1
df_biodiv_all = df_nonzero[
    (df_nonzero['binary_label_biodiversity_impact'] == 1) &
    (df_nonzero['No_Biodiv'] != 1)
].copy()

bio_all_per_year = (
    df_biodiv_all.groupby("Year")["Count"].sum()
      .reindex(YEARS)
      .fillna(0)
)

# 2) DISBURSEMENT LINES & PIVOTS
# Rio Marker disbursement lines (unweighted & weighted)
_df_rm = df[df["USD_Disbursement"] > 0].copy()

rm_map_unweighted = {2: 1.0, 1: 1.0, 0: 0}
rm_map_weighted   = {2: 1.0, 1: 0.4, 0: 0}

_df_rm["rm_w_unweighted"] = _df_rm[RIO_MARKER_COL].map(rm_map_unweighted).fillna(0)
_df_rm["rm_w_weighted"]   = _df_rm[RIO_MARKER_COL].map(rm_map_weighted).fillna(0)

_df_rm["disb_unweighted"] = _df_rm["USD_Disbursement"] * _df_rm["rm_w_unweighted"]
_df_rm["disb_weighted"]   = _df_rm["USD_Disbursement"] * _df_rm["rm_w_weighted"]

line_unweighted = (
    _df_rm.groupby("Year")["disb_unweighted"].sum().reindex(YEARS).fillna(0)
)
line_weighted = (
    _df_rm.groupby("Year")["disb_weighted"].sum().reindex(YEARS).fillna(0)
)

# Disbursement subset for stacked bars
df_pos = df[
    (df['USD_Disbursement'] > 0) &
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1)
].copy()

# Donor folding (Bilateral / Multilateral / Private / Other)
df_pos["Bi_Multi"] = pd.to_numeric(df_pos["Bi_Multi"], errors="coerce")
donor_fold_map = {
    1: "Bilateral", 7: "Bilateral", 3: "Bilateral", 8: "Bilateral",
    4: "Multilateral",
    6: "Private",
    5: "Other"
}
order_donor_labels = ["Bilateral", "Multilateral", "Private", "Other"]
df_pos["DonorLabel"] = df_pos["Bi_Multi"].map(donor_fold_map)

donor_pivot = (
    df_pos[df_pos["DonorLabel"].notna()]
      .pivot_table(index="Year", columns="DonorLabel",
                   values="USD_Disbursement", aggfunc="sum")
      .reindex(YEARS)
      .fillna(0)
)
donor_pivot = donor_pivot[[c for c in order_donor_labels if c in donor_pivot.columns]]

donor_color_map = {
    "Bilateral":    "#d9d9d9",
    "Multilateral": "#a6a6a6",
    "Private":      "#737373",
    "Other":        "#1a1a1a",
}
donor_colors_used = [donor_color_map[c] for c in donor_pivot.columns]

# Flow type split
flow_order = [
    "ODA Grants",
    "ODA Loans",
    "Other Official Flows (non Export Credit)",
    "Private Development Finance",
    "Equity Investment"
]
flow_pivot = (
    df_pos[df_pos["FlowName"].isin(flow_order)]
      .pivot_table(index="Year", columns="FlowName", values="USD_Disbursement", aggfunc="sum")
      .reindex(YEARS)
      .fillna(0)
)
flow_cols = [c for c in flow_order if c in flow_pivot.columns]
flow_pivot = flow_pivot[flow_cols]

flow_color_map = {
    "ODA Grants": "#d9d9d9",
    "ODA Loans": "#bfbfbf",
    "Other Official Flows (non Export Credit)": "#999999",
    "Private Development Finance": "#595959",
    "Equity Investment": "#1a1a1a",
}
flow_colors_used = [flow_color_map[c] for c in flow_pivot.columns]

# Goal (merged), Instrument, Ecosystem for disbursement bars
act_disb_merged = merge_act_groups(act_disb)


# ___________________________________________________________
#  MAIN AND SI: Goal_Eco_Impl_Project_Count_Finance
# ___________________________________________________________

fig1, axs1 = plt.subplots(2, 3, figsize=(18, 12))

# Unpack axes for clarity
ax_goal_cnt_1, ax_impl_cnt_1, ax_eco_cnt_1 = axs1[0, 0], axs1[0, 1], axs1[0, 2]
ax_goal_disb_1, ax_impl_disb_1, ax_eco_disb_1 = axs1[1, 0], axs1[1, 1], axs1[1, 2]

# ---------- TOP ROW: PROJECT COUNTS (Goal, Instrument, Ecosystem) ----------
for ax, data, title, colors, col_order, labels_dict in zip(
        [ax_goal_cnt_1, ax_impl_cnt_1, ax_eco_cnt_1],
        [act_merged_count_f, impl_count_f,  eco_count_f],
        ['Goal',            'Instrument',   'Ecosystem'],
        [act_merged_colors, impl_colors,    eco_colors],
        [act_merged_order,  impl_order,     eco_order],
        [act_merged_labels_nobreak, impl_labels_nobreak, eco_labels_nobreak]
):
    # ensure correct columns & YEAR index
    valid_cols = [col for col in col_order if col in data.columns]
    df_plot = data.loc[:, valid_cols].copy()
    df_plot = df_plot.reindex(YEARS).fillna(0)

    color_map = [colors[col_order.index(col)] for col in valid_cols]
    renamed_cols = [labels_dict.get(col, col) for col in valid_cols]
    df_plot.columns = renamed_cols

    # AREA chart for counts
    df_plot.plot.area(ax=ax, stacked=True, color=color_map)

    ax.margins(x=0.02, y=0)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    ax.set_xlabel(" ", fontsize=LABEL_FONTSIZE)
    ax.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)

    # Only left-most plot gets y-label
    if ax is ax_goal_cnt_1:
        ax.set_ylabel("Project Count", fontsize=14)
    else:
        ax.set_ylabel("")

    # X-axis ticks (every 5 years)
    ax.set_xticks([YEARS[i] for i in xtick_positions])
    ax.set_xticklabels([xtick_labels[i] for i in range(len(xtick_positions))], rotation=0)

# ---------- BOTTOM ROW: DISBURSEMENTS (Goal, Instrument, Ecosystem) ----------
for ax, data_in, title, colors, col_order, labels_dict in [
    (ax_goal_disb_1,  merge_act_groups(act_disb), 'Goal',       act_merged_colors, act_merged_order, act_merged_labels_nobreak),
    (ax_impl_disb_1,  impl_disb,                  'Instrument', impl_colors,       impl_order,       impl_labels_nobreak),
    (ax_eco_disb_1,   eco_disb,                   'Ecosystem',  eco_colors,        eco_order,        eco_labels_nobreak),
]:

    df_plot, color_list = _prep_stacked(data_in, col_order, colors, labels_dict)

    # AREA chart instead of bars
    df_plot.plot.area(ax=ax, stacked=True, color=color_list)

    ax.margins(x=0.02, y=0)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    ax.set_xlabel(" ", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(0, DISB_YMAX)

    # Only left-most plot gets y-label
    if ax is ax_goal_disb_1:
        ax.set_ylabel("Disbursements [USD bn]", fontsize=14)
    else:
        ax.set_ylabel("")

    # X-axis ticks (every 5 years)
    ax.set_xticks([YEARS[i] for i in xtick_positions])
    ax.set_xticklabels(xtick_labels, rotation=0)

    handles_b, labels_b = ax.get_legend_handles_labels()
    ax.legend(handles_b, labels_b, loc='upper left', fontsize=LEGEND_FONTSIZE)

plt.tight_layout()
output_path= "outputs/MAIN_SI_Goal_Eco_Impl_Project_Count_Finance.png"
plt.savefig(output_path, dpi=300)
plt.savefig(output_path, dpi=300)
plt.show()
print(f"Main + SI Figure: Saved Goal / Instrument / Ecosystem project count + disbursement area charts to {output_path}")

# -------------------------------------------------------------
# SI: BIODIV CUBE DONOR FLOWTYPE
# -------------------------------------------------------------

# ---------- Helper data for FIGURE 2 ----------

# 1) ODF-BiodivBERT project count per year (you already computed bio_all_per_year)
bio_all_count_year = bio_all_per_year.reindex(YEARS).fillna(0)

# 1) Base filter: only positive disbursements AND biodiversity subset
df_pos = df[
    (df['USD_Disbursement'] > 0) &
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1)
].copy()

# 2) Project COUNT by Donor type (Biodiv Cube filter: df_pos)
df_pos_donor = df_pos.copy()
df_pos_donor["Bi_Multi"] = pd.to_numeric(df_pos_donor["Bi_Multi"], errors="coerce")
df_pos_donor["DonorLabel"] = df_pos_donor["Bi_Multi"].map(donor_fold_map)

donor_count_pivot = (
    df_pos_donor[df_pos_donor["DonorLabel"].notna()]
        .groupby(["Year", "DonorLabel"])
        .size()
        .unstack("DonorLabel")
        .reindex(YEARS)
        .fillna(0)
)

# ensure consistent donor column order
donor_count_pivot = donor_count_pivot[[c for c in order_donor_labels if c in donor_count_pivot.columns]]
donor_count_colors = [donor_color_map[c] for c in donor_count_pivot.columns]

# Donor-type disbursements (Biodiv Cube filter: df_pos)
donor_pivot = (
    df_pos_donor[df_pos_donor["DonorLabel"].notna()]
        .groupby(["Year", "DonorLabel"])["USD_Disbursement"]
        .sum()
        .unstack("DonorLabel")
        .reindex(YEARS)
        .fillna(0)
)

# 3) Project COUNT by Finance flow type (Biodiv Cube filter: df_pos)
df_pos_flow = df_pos.copy()
flow_count_pivot = (
    df_pos_flow[df_pos_flow["FlowName"].isin(flow_order)]
        .groupby(["Year", "FlowName"])
        .size()
        .unstack("FlowName")
        .reindex(YEARS)
        .fillna(0)
)
flow_count_pivot = flow_count_pivot[[c for c in flow_order if c in flow_count_pivot.columns]]
flow_count_colors = [flow_color_map[c] for c in flow_count_pivot.columns]

# Flow-type disbursements (Biodiv Cube filter: df_pos)
flow_pivot = (
    df_pos_flow[df_pos_flow["FlowName"].isin(flow_order)]
        .groupby(["Year", "FlowName"])["USD_Disbursement"]
        .sum()
        .unstack("FlowName")
        .reindex(YEARS)
        .fillna(0)
)

# 4) Total ODF-BiodivBERT disbursements per year (using df_pos filter)
biodiv_disb_year = (
    df_pos.groupby("Year")["USD_Disbursement"]
         .sum()
         .reindex(YEARS)
         .fillna(0)
)

# ---------- Build FIGURE 2 ----------

fig2, axs2 = plt.subplots(2, 3, figsize=(18, 12))

# Unpack axes
ax_allbiodiv_cnt_2, ax_donor_cnt_2, ax_flow_cnt_2 = axs2[0, 0], axs2[0, 1], axs2[0, 2]
ax_allbiodiv_disb_2, ax_donor_disb_2, ax_flow_disb_2 = axs2[1, 0], axs2[1, 1], axs2[1, 2]

# ---------- TOP LEFT: ODF-BiodivBERT project count + Rio COUNT line ----------
ax = ax_allbiodiv_cnt_2
ax.fill_between(
    YEARS,
    bio_all_count_year.values,
    color='lightgrey',
    alpha=0.9,
    label='Biodiv Cube'
)
ax.plot(
    YEARS,
    rio_count_year.reindex(YEARS).fillna(0).values,
    linestyle='--',
    color='red',
    linewidth=3,
    label='Biodiversity Rio Marker (1&2)'
)

ax.margins(x=0.02, y=0)
ax.set_title("Biodiv Cube (Total)", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
ax.set_ylabel("Project Count", fontsize=14)
ax.set_xlabel(" ", fontsize=LABEL_FONTSIZE)
ax.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)

ax.set_xticks([YEARS[i] for i in xtick_positions])
ax.set_xticklabels(xtick_labels, rotation=0)

# ---------- TOP MIDDLE: Project COUNT by Donor type (AREA) ----------
ax = ax_donor_cnt_2
donor_count_pivot.plot.area(ax=ax, stacked=True, color=donor_count_colors)

ax.margins(x=0.02, y=0)
ax.set_title("Donor Type", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
ax.set_xlabel(" ", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("")  # only leftmost column has y-label
ax.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)

ax.set_xticks([YEARS[i] for i in xtick_positions])
ax.set_xticklabels(xtick_labels, rotation=0)

# ---------- TOP RIGHT: Project COUNT by Finance Flow type (AREA) ----------
ax = ax_flow_cnt_2
flow_count_pivot.plot.area(ax=ax, stacked=True, color=flow_count_colors)

ax.margins(x=0.02, y=0)
ax.set_title("Finance Flow Type", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
ax.set_xlabel(" ", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("")  # only leftmost column has y-label
ax.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)

ax.set_xticks([YEARS[i] for i in xtick_positions])
ax.set_xticklabels(xtick_labels, rotation=0)

# ---------- Unify Y-axis for TOP ROW (project counts) ----------

# Total project count per year (left chart)
max_total = bio_all_count_year.max()

# Stacked donor counts per year (middle chart)
max_donor = donor_count_pivot.sum(axis=1).max()

# Stacked flow counts per year (right chart)
max_flow = flow_count_pivot.sum(axis=1).max()

# Choose a common upper limit with a little headroom
max_count = max(max_total, max_donor, max_flow)
# Optional: round up to nearest 10 for nicer axis

ymax_counts = math.ceil(max_count / 10) * 10

for ax in [ax_allbiodiv_cnt_2, ax_donor_cnt_2, ax_flow_cnt_2]:
    ax.set_ylim(0, ymax_counts)


# ---------- BOTTOM LEFT: ODF-BiodivBERT Disbursements + Rio DISBURSEMENT lines ----------
ax = ax_allbiodiv_disb_2

# AREA (single) for total ODF-BiodivBERT disbursement
ax.fill_between(
    YEARS,
    biodiv_disb_year.values,
    color='lightgrey',
    alpha=0.9,
    label='Biodiv Cube'
)

# Rio disbursement lines
ax.plot(
    YEARS,
    line_unweighted.reindex(YEARS).values,
    linestyle='--',
    linewidth=4,
    color='red',
    label='Biodiversity Rio Marker (1&2, unweighted)'
)
ax.plot(
    YEARS,
    line_weighted.reindex(YEARS).values,
    linestyle='-.',
    linewidth=4,
    color='#FF9999',
    label='Biodiversity Rio Marker (1&2, weighted)'
)

ax.margins(x=0.02, y=0)
ax.set_title("Biodiv Cube (Total)", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
ax.set_ylabel("Disbursements [USD bn]", fontsize=14)
ax.set_xlabel(" ", fontsize=LABEL_FONTSIZE)
ax.set_ylim(0, DISB_YMAX)

ax.set_xticks([YEARS[i] for i in xtick_positions])
ax.set_xticklabels(xtick_labels, rotation=0)

handles, labels_ = ax.get_legend_handles_labels()
ax.legend(handles, labels_, loc='upper left', fontsize=LEGEND_FONTSIZE)

# ---------- BOTTOM MIDDLE: Donor-type Disbursements (AREA) ----------
ax = ax_donor_disb_2
donor_plot_area = donor_pivot

donor_plot_area.plot.area(ax=ax, stacked=True, color=donor_colors_used)

ax.margins(x=0.02, y=0)
ax.set_title("Donor Type", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
ax.set_ylabel("")
ax.set_xlabel(" ", fontsize=LABEL_FONTSIZE)
ax.set_ylim(0, DISB_YMAX)

ax.set_xticks([YEARS[i] for i in xtick_positions])
ax.set_xticklabels(xtick_labels, rotation=0)

handles_b, labels_b = ax.get_legend_handles_labels()
ax.legend(handles_b, labels_b, loc='upper left', fontsize=LEGEND_FONTSIZE)

# ---------- BOTTOM RIGHT: Finance Flow-type Disbursements (AREA) ----------
ax = ax_flow_disb_2
flow_plot_area = flow_pivot

flow_plot_area.plot.area(ax=ax, stacked=True, color=flow_colors_used)

ax.margins(x=0.02, y=0)
ax.set_title("Finance Flow Type", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
ax.set_ylabel("")
ax.set_xlabel(" ", fontsize=LABEL_FONTSIZE)
ax.set_ylim(0, DISB_YMAX)

ax.set_xticks([YEARS[i] for i in xtick_positions])
ax.set_xticklabels(xtick_labels, rotation=0)

handles_f, labels_f = ax.get_legend_handles_labels()
ax.legend(handles_f, labels_f, loc='upper left', fontsize=LEGEND_FONTSIZE)

# ---------- Finalize FIGURE 2 ----------
plt.tight_layout()
output_path= "outputs/SI_Biodiv_Cube_Donor_Flowtype.png"
plt.savefig(output_path, dpi=300)
plt.savefig(output_path, dpi=300)
plt.show()
print(f"SI Figure: Saved Biodiv Cube donor / flow type project count + disbursement area charts to {output_path}")


# EXPORT XLSX: Yearly disbursement values + donor/flow breakdown
# 1) Summary: Biodiv Cube + Rio markers (bottom-left chart)
# Ensure Rio marker series are aligned with YEARS
rio_unweighted_year = line_unweighted.reindex(YEARS)
rio_weighted_year   = line_weighted.reindex(YEARS)

disb_export_df = pd.DataFrame({
    "Year": YEARS,
    "Biodiv Cube disbursements [USD bn]": biodiv_disb_year.values,
    "Rio Marker (1&2, unweighted) [USD bn]": rio_unweighted_year.values,
    "Rio Marker (1&2, weighted) [USD bn]": rio_weighted_year.values,
})

donor_disb_export_df = donor_pivot.copy()
donor_disb_export_df.insert(0, "Year", YEARS)

flow_disb_export_df = flow_pivot.copy()
flow_disb_export_df.insert(0, "Year", YEARS)

# 4) Write all three tables into a single Excel file (three sheets)

output_xlsx_path = "outputs/Yearly_Disbursements_BiodivCube_and_RioMarkers.xlsx"

with pd.ExcelWriter(output_xlsx_path, engine="xlsxwriter") as writer:
    # Sheet 1: Summary lines
    disb_export_df.to_excel(writer, sheet_name="Summary_Disbursements", index=False)

    # Sheet 2: Biodiv Cube by donor type
    donor_disb_export_df.to_excel(writer, sheet_name="BiodivCube_by_DonorType", index=False)

    # Sheet 3: Biodiv Cube by flow type
    flow_disb_export_df.to_excel(writer, sheet_name="BiodivCube_by_FlowType", index=False)

print(f"SI Table: Exported yearly disbursement tables to {output_xlsx_path}")


# ================================================
# MAIN: FIGURE 2 2030 GBF TARGET PROJECTIONS
# ================================================

# Extended year range to 2030
YEARS_EXT = list(range(2000, 2031))

# Reindex Biodiv Cube disbursement and Rio-marker lines to extended years
biodiv_disb_year_ext = (
    biodiv_disb_year
    .reindex(YEARS_EXT)
)

line_unweighted_ext = (
    line_unweighted
    .reindex(YEARS_EXT)
)

line_weighted_ext = (
    line_weighted
    .reindex(YEARS_EXT)
)

# Get 2023 value for projection starting point
last_data_year = 2023
y_2023 = biodiv_disb_year.loc[last_data_year]

# Target values for 2030 (in USD bn)
target_low_2030  = 17.0
target_high_2030 = 25.2

# Projection x-values
x_proj = [last_data_year, 2031]
y_proj_low  = [y_2023, target_low_2030]
y_proj_high = [y_2023, target_high_2030]

# Extended x-ticks every 5 years
xtick_positions_ext = [i for i, y in enumerate(YEARS_EXT) if y % 5 == 0]
xtick_labels_ext    = [str(YEARS_EXT[i]) for i in xtick_positions_ext]

# Figure
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

ax.plot(
    YEARS_EXT,
    biodiv_disb_year_ext.values,
    linewidth=4,
    color='lightgrey',
    label='Biodiv Cube'
)

# ---- Rio disbursement lines (stop at last data year) ----
ax.plot(
    YEARS_EXT,
    line_unweighted_ext.values,
    #linestyle='--',
    linewidth=4,
    color='red',
    label='Biodiversity Rio Marker (1&2, unweighted)'
)

ax.plot(
    YEARS_EXT,
    line_weighted_ext.values,
    #linestyle='-.',
    linewidth=4,
    color='#FF9999',
    label='Biodiversity Rio Marker (1&2, weighted)'
)

# ---- GBF horizontal line at 30 bn ----
gbf_target = 30.0  # USD bn
ax.axhline(
    y=gbf_target,
    color='darkgreen',
    linewidth=4
)

# Text label on/near the GBF line
ax.text(
    x=2000.3,
    y=gbf_target + 0.5,
    s="2030 Target, Global Biodiversity Framework (USD 30 bn)",
    color='darkgreen',
    fontsize=14,
    fontweight='bold',
    va='bottom',
    ha='left'
)

# ---- Development lines 2023–2030 ----
ax.plot(
    x_proj,
    y_proj_low,
    linestyle='--',
    linewidth=4,
    color='lightgrey',
    label='Trajectory to 15.7 bn (2030)'
)

ax.plot(
    x_proj,
    y_proj_high,
    linestyle='--',
    linewidth=4,
    color='lightgrey',
    label='Trajectory to 17.5 bn (2030)'
)

# ---- Formatting ----
ax.margins(x=0.02, y=0)
ax.set_ylabel("Disbursements [USD bn]", fontsize=14)
ax.set_xlabel(" ", fontsize=LABEL_FONTSIZE)
ax.set_ylim(0, 33)
ax.set_xticks([YEARS_EXT[i] for i in xtick_positions_ext])
ax.set_xticklabels(xtick_labels_ext, rotation=0)


plt.tight_layout()
output_path= "outputs/MAIN_2030_GBF_Target_Projections.png"
plt.savefig(output_path, dpi=300)
plt.savefig("outputs/MAIN_2030_GBF_Target_Projections.pdf", dpi=300)
plt.show()
print(f"Main Figure: Saved 2030 GBF target projection chart to {output_path} and .pdf")


# ---------------------------------------
# Disbursement tables for Donor Type & Flow Type
# ---------------------------------------

# --- helpers (same spirit as prep_disb_table) ---
def _coerce_year_index(df_like, years=None, round_ndigits=3):
    """Ensure numeric year index, sorted; optionally reindex to a fixed range; add Total; round."""
    tbl = df_like.copy()
    yr = pd.to_numeric(pd.Index(tbl.index).astype(str), errors="coerce")
    mask = ~pd.isna(yr)
    tbl = tbl.iloc[mask].copy()
    tbl.index = yr[mask].astype(int)
    tbl = tbl.sort_index()
    if years is not None:
        tbl = tbl.reindex(years)
    tbl["Total"] = tbl.sum(axis=1, min_count=1)
    return tbl.round(round_ndigits)

def _wide_to_long(wide_df, value_name="USD bn"):
    """Convert wide (years x categories + Total) to long (Year, Category, value)."""
    return (wide_df.reset_index(names="Year")
                  .melt(id_vars="Year", var_name="Category", value_name=value_name))


# 0 Config from  plotting block
# --- Donor folding (exclusive labels as provided) ---
donor_fold_map = {
    1: "Bilateral", 7: "Bilateral", 3: "Bilateral", 8: "Bilateral",
    4: "Multilateral",
    6: "Private",
    5: "Other"
}
order_donor_labels = ["Bilateral", "Multilateral", "Private", "Other"]

# --- Flow order (exclusive labels as provided) ---
flow_order  = [
    "ODA Grants",
    "ODA Loans",
    "Other Official Flows (non Export Credit)",
    "Private Development Finance",
    "Equity Investment"
]

# 1) Base filter (same as for bars)

df_pos = df[
    (df['USD_Disbursement'] > 0) &
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1)
].copy()

# Make sure Bi_Multi is numeric before mapping
df_pos["Bi_Multi"] = pd.to_numeric(df_pos["Bi_Multi"], errors="coerce")


# 2) DONOR TYPE table (wide + long)

# Add donor label
df_pos["DonorLabel"] = df_pos["Bi_Multi"].map(donor_fold_map)

# Pivot to wide (Year x DonorLabel)
donor_pivot_raw = (
    df_pos[df_pos["DonorLabel"].notna()]
      .pivot_table(index="Year", columns="DonorLabel",
                   values="USD_Disbursement", aggfunc="sum")
)

# Keep only desired order, if present
donor_cols = [c for c in order_donor_labels if c in donor_pivot_raw.columns]
donor_pivot_raw = donor_pivot_raw[donor_cols]

# Coerce year index, reindex to YEARS, add Total, round
donor_wide = _coerce_year_index(donor_pivot_raw, years=YEARS, round_ndigits=3)

# Long format
donor_long = _wide_to_long(donor_wide, value_name="USD bn")


# 3) FLOW TYPE table (wide + long)

flow_pivot_raw = (
    df_pos[df_pos["FlowName"].isin(flow_order)]
      .pivot_table(index="Year", columns="FlowName",
                   values="USD_Disbursement", aggfunc="sum")
)

# Enforce  flow order
flow_cols = [c for c in flow_order if c in flow_pivot_raw.columns]
flow_pivot_raw = flow_pivot_raw[flow_cols]

# Coerce year index, reindex to YEARS, add Total, round
flow_wide = _coerce_year_index(flow_pivot_raw, years=YEARS, round_ndigits=3)

# Long format
flow_long = _wide_to_long(flow_wide, value_name="USD bn")

# 4) Write to Excel
os.makedirs(
    "./outputs", exist_ok=True)
out_xlsx = "outputs/SI_disbursement_tables_donor_flowtype.xlsx"

with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    donor_wide.to_excel(writer, sheet_name="Donor (wide)")
    donor_long.to_excel(writer, sheet_name="Donor (long)", index=False)
    flow_wide.to_excel(writer, sheet_name="Flow (wide)")
    flow_long.to_excel(writer, sheet_name="Flow (long)", index=False)

print(f"SI Table: Exported donor and flow disbursement tables {out_xlsx}")

# ---------------------------------------
# Project-count tables for Donor Type & Flow Type
# ---------------------------------------

# --- helpers ---
def _coerce_year_index(df_like, years=None, round_ndigits=None):
    """Ensure numeric year index, sorted; optionally reindex; add Total; optional rounding."""
    tbl = df_like.copy()
    yr = pd.to_numeric(pd.Index(tbl.index).astype(str), errors="coerce")
    mask = ~pd.isna(yr)
    tbl = tbl.iloc[mask].copy()
    tbl.index = yr[mask].astype(int)
    tbl = tbl.sort_index()
    if years is not None:
        tbl = tbl.reindex(years)
    tbl["Total"] = tbl.sum(axis=1, min_count=1)
    if round_ndigits is not None:
        tbl = tbl.round(round_ndigits)
    return tbl

def _wide_to_long(wide_df, value_name="Project Count"):
    return (wide_df.reset_index(names="Year")
                  .melt(id_vars="Year", var_name="Category", value_name=value_name))

# --- label configs (exclusive) ---
donor_fold_map = {
    1: "Bilateral", 7: "Bilateral", 3: "Bilateral", 8: "Bilateral",
    4: "Multilateral",
    6: "Private",
    5: "Other"
}
order_donor_labels = ["Bilateral", "Multilateral", "Private", "Other"]

flow_order  = [
    "ODA Grants",
    "ODA Loans",
    "Other Official Flows (non Export Credit)",
    "Private Development Finance",
    "Equity Investment"
]

# 1) Base filter for counts (ALIGN WITH DISBURSEMENT)

df_pos_counts = df[
    (df['USD_Disbursement'] > 0) &
    (df['binary_label_biodiversity_impact'] == 1) &
    (df['No_Biodiv'] != 1)
].copy()

# 2) DONOR TYPE project counts (aligned)

for _df, suffix in [(df_pos_counts, "")]:
    _df["Bi_Multi"] = pd.to_numeric(_df["Bi_Multi"], errors="coerce")
    _df["DonorLabel"] = _df["Bi_Multi"].map(donor_fold_map)

    donor_ct_raw = (
        _df[_df["DonorLabel"].notna()]
          .assign(_ones=1)
          .pivot_table(index="Year", columns="DonorLabel", values="_ones", aggfunc="sum")
    )
    donor_cols = [c for c in order_donor_labels if c in donor_ct_raw.columns]
    donor_ct_raw = donor_ct_raw[donor_cols]

    donor_count_wide = _coerce_year_index(donor_ct_raw, years=YEARS)
    donor_count_long = _wide_to_long(donor_count_wide)

# 3) FLOW TYPE project counts (aligned)

for _df, suffix in [(df_pos_counts, "")]:
    flow_ct_raw = (
        _df[_df["FlowName"].isin(flow_order)]
          .assign(_ones=1)
          .pivot_table(index="Year", columns="FlowName", values="_ones", aggfunc="sum")
    )
    flow_cols = [c for c in flow_order if c in flow_ct_raw.columns]
    flow_ct_raw = flow_ct_raw[flow_cols]

    flow_count_wide = _coerce_year_index(flow_ct_raw, years=YEARS)
    flow_count_long = _wide_to_long(flow_count_wide)


# 5) Write to Excel (aligned)

os.makedirs(
    "./outputs", exist_ok=True)
out_xlsx = "outputs/SI_project_count_tables_donor_flowtype.xlsx"

with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    donor_count_wide.to_excel(writer, sheet_name="Donor Count (wide)")
    donor_count_long.to_excel(writer, sheet_name="Donor Count (long)", index=False)
    flow_count_wide.to_excel(writer,  sheet_name="Flow Count (wide)")
    flow_count_long.to_excel(writer,  sheet_name="Flow Count (long)", index=False)

print(f"SI Table: Exported project count tables for donor and finance flow types SI to {out_xlsx}")


# ------------------------------------------------------------
# MAIN 3D MATRIX FIGURE
# ------------------------------------------------------------

# 1. Load data (long format)

file_path = "outputs/act_impl_eco_3D_annual_avg_disbursements.xlsx"
df1 = pd.read_excel(file_path)

# Column names (adjust value_col if needed)
act_col = "Act_Class"
eco_col = "Eco_Class"
impl_col = "Impl_Class"
value_col = "Annual_Avg"

df1 = df1[[act_col, eco_col, impl_col, value_col]]

# 2. Define implementation classes and colors

impl_order = ['Impl_Regul', 'Impl_Know', 'Impl_Infra', 'Impl_Undef']
impl_colors = ['#A6F453', '#CBD94A', '#aab63d', 'lightgrey']
impl_color_map = dict(zip(impl_order, impl_colors))

actions = ["Act_Pollut", "Act_Invasiv", "Act_SustResMgmt", "Act_Protect_Resto", "Act_Undef"]
ecosystems = ["Eco_CropGrass", "Eco_Forest", "Eco_SeaWater", "Eco_UrbInd", "Eco_Undef"]

# Quadrant positions inside each big cell
quadrant_positions = {
    impl_order[0]: (0, 1),  # top-left
    impl_order[1]: (1, 1),  # top-right
    impl_order[2]: (0, 0),  # bottom-left
    impl_order[3]: (1, 0),  # bottom-right
}


# 3. Per-implementation maxima + normalization (per Impl_Class)
# Max disbursement for each implementation category
impl_max = df1.groupby(impl_col)[value_col].max().to_dict()

# PowerNorm with gamma < 1 so lower values are more visible
gamma = 0.5
impl_norms = {
    impl: mcolors.PowerNorm(
        gamma=gamma,
        vmin=0,
        vmax=impl_max.get(impl, 1) if impl_max.get(impl, 0) > 0 else 1
    )
    for impl in impl_order
}

# Build a colormap (white -> base color) for each Impl_Class
impl_cmaps = {
    impl: mcolors.LinearSegmentedColormap.from_list(
        name=impl,
        colors=["#ffffff", impl_color_map[impl]]
    )
    for impl in impl_order
}


# 4. Create main figure and axes
fig, ax = plt.subplots(figsize=(1.4 * len(actions), 1.0 * len(ecosystems)))


# 5. Draw subcells with values
for yi, eco in enumerate(ecosystems):
    for xi, act in enumerate(actions):

        # Subset for this Act × Eco
        sub = df1[(df1[act_col] == act) & (df1[eco_col] == eco)]

        # Aggregate disbursement per Impl_Class
        values = {
            impl: sub.loc[sub[impl_col] == impl, value_col].sum()
            for impl in impl_order
        }

        for impl in impl_order:
            v = values[impl]
            qx, qy = quadrant_positions[impl]
            local_max = impl_max.get(impl, 0)
            local_norm = impl_norms[impl]

            # Normalize to [0,1] within this implementation category
            if v <= 0 or np.isnan(v) or local_max <= 0:
                norm_v = 0.0
            else:
                norm_v = local_norm(v)  # PowerNorm already returns 0–1

            color = impl_cmaps[impl](norm_v)

            # Subcell geometry: each big cell is 1×1, subcells are 0.5×0.5
            cell_x = xi + 0.5 * qx
            cell_y = yi + 0.5 * qy

            ax.add_patch(
                Rectangle(
                    (cell_x, cell_y),
                    0.5, 0.5,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5
                )
            )

            # Value label with thousands separator
            if v >= 0.01:
                ax.text(
                    cell_x + 0.25,
                    cell_y + 0.25,
                    f"{v:,.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                )

            elif v > 0:
                ax.text(
                    cell_x + 0.25,
                    cell_y + 0.25,
                    "<0.01",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                )


# 6. Format axes
ax.set_xlim(0, len(actions))
ax.set_ylim(0, len(ecosystems))
ax.invert_yaxis()

ax.set_xticks(np.arange(len(actions)) + 0.5)
ax.set_yticks(np.arange(len(ecosystems)) + 0.5)
ax.set_xticklabels(actions, rotation=45, ha="right")
ax.set_yticklabels(ecosystems)

ax.set_xlabel(act_col)
ax.set_ylabel(eco_col)
ax.set_title("Act_Class × Eco_Class × Impl_Class\nAnnual avg disbursements)")

# Draw grid for big cells
for x in range(len(actions) + 1):
    ax.axvline(x, color="black", linewidth=0.5)
for y in range(len(ecosystems) + 1):
    ax.axhline(y, color="black", linewidth=0.5)

# Leave space on the right for colorbars
plt.tight_layout(rect=[0, 0, 0.8, 1])


# 7. Colorbar legend (outside, right; one per Impl_Class)
bar_height = 0.03
gap = 0.01
start_y = 0.08

for i, impl in enumerate(impl_order):
    local_max = impl_max.get(impl, 0)
    if local_max <= 0:
        continue

plt.tight_layout()

#plt.savefig("Act_Eco_Impl_Heatmap_relative_colors.png", dpi=300)

#Relative to Overall Maximum Value (Single Color Scale)
# 2. Setup classes and colors
impl_order = ['Impl_Regul', 'Impl_Know', 'Impl_Infra', 'Impl_Undef']
impl_colors = ['#A6F453', '#CBD94A', '#aab63d', 'lightgrey']
impl_color_map = dict(zip(impl_order, impl_colors))

actions = ["Act_Pollut", "Act_Invasiv", "Act_SustResMgmt", "Act_Protect_Resto", "Act_Undef"]
ecosystems = ["Eco_CropGrass", "Eco_Forest", "Eco_SeaWater", "Eco_UrbInd", "Eco_Undef"]

# Quadrant layout in each big cell
quadrant_positions = {
    impl_order[0]: (0, 1),  # top-left
    impl_order[1]: (1, 1),  # top-right
    impl_order[2]: (0, 0),  # bottom-left
    impl_order[3]: (1, 0),  # bottom-right
}

# 3. Global max + normalization (gamma to boost low values)
global_max = df1[value_col].max()

# PowerNorm with gamma < 1 brightens low values
gamma = 0.5
norm = mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max)

# Build a colormap (white -> base color) for each Impl_Class
impl_cmaps = {
    impl: mcolors.LinearSegmentedColormap.from_list(
        name=impl,
        colors=["#ffffff", base_color]
    )
    for impl, base_color in impl_color_map.items()
}

# 4. Create figure and axes
fig, ax = plt.subplots(figsize=(1.4 * len(actions), 1.0 * len(ecosystems)))

# 5. Draw subcells with values
for yi, eco in enumerate(ecosystems):
    for xi, act in enumerate(actions):

        sub = df1[(df1[act_col] == act) & (df1[eco_col] == eco)]

        # Aggregate disbursement by Impl_Class
        values = {
            impl: sub.loc[sub[impl_col] == impl, value_col].sum()
            for impl in impl_order
        }

        for impl in impl_order:
            v = values[impl]
            qx, qy = quadrant_positions[impl]

            # Map disbursement -> [0,1] via PowerNorm
            if v <= 0 or np.isnan(v) or global_max == 0:
                norm_v = 0.0
            else:
                norm_v = norm(v)  # already in [0,1] with gamma correction

            color = impl_cmaps[impl](norm_v)

            # Subcell geometry
            cell_x = xi + 0.5 * qx
            cell_y = yi + 0.5 * qy

            rect = Rectangle(
                (cell_x, cell_y),
                0.5, 0.5,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5
            )
            ax.add_patch(rect)

            # Value label with thousands separator
            if v >= 0.01:
                ax.text(
                    cell_x + 0.25,
                    cell_y + 0.25,
                    f"{v:,.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                )

            elif v > 0:
                ax.text(
                    cell_x + 0.25,
                    cell_y + 0.25,
                    "<0.01",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                )


# 6. Format axes
ax.set_xlim(0, len(actions))
ax.set_ylim(0, len(ecosystems))
ax.invert_yaxis()

ax.set_xticks(np.arange(len(actions)) + 0.5)
ax.set_yticks(np.arange(len(ecosystems)) + 0.5)
ax.set_xticklabels(actions, rotation=45, ha="right")
ax.set_yticklabels(ecosystems)

ax.set_xlabel(act_col)
ax.set_ylabel(eco_col)
ax.set_title("Act_Class × Eco_Class × Impl_Class\nAnnual avg disbursements")

# Grid for big cells
for x in range(len(actions) + 1):
    ax.axvline(x, color="black", linewidth=0.5)
for y in range(len(ecosystems) + 1):
    ax.axhline(y, color="black", linewidth=0.5)


# 7. Colormap legend (0–100%) + max value
bar_height = 0.03  # height of each horizontal bar
gap = 0.01  # vertical gap between bars
start_y = 0.08  # starting y-position of the lowest bar (figure coords)

for i, impl in enumerate(impl_order):
    local_max = impl_max.get(impl, 0)
    if local_max <= 0:
        continue

    cax = fig.add_axes([
        1,
        start_y + i * (bar_height + gap),
        0.12,
        bar_height
    ])

    local_norm = impl_norms[impl]
    sm = plt.cm.ScalarMappable(norm=local_norm, cmap=impl_cmaps[impl])
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")

    # Remove ticks and labels
    cbar.ax.set_xticks([])
    cbar.ax.set_yticks([])
    cbar.ax.tick_params(left=False, right=False, top=False, bottom=False,
                        labelleft=False, labelright=False, labeltop=False, labelbottom=False)

    # Remove colorbar frame (spines)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    # Add text label to the right of the colorbar
    fig.text(
        1.15,
        start_y + i * (bar_height + gap) + bar_height / 2,
        f"{impl} (max: {local_max:,.0f})",
        ha="left",
        va="center",
        fontsize=8
    )

plt.tight_layout()
output_path= "outputs/Act_Eco_Impl_3DHeatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig("outputs/Act_Eco_Impl_3DHeatmap.pdf", dpi=300, bbox_inches='tight')
plt.show()
print(f"Main Figure: Exported 3D Heatmap to {output_path} and .pdf")