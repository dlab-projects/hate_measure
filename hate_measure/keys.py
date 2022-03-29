# Hate speech construct items
items = [
    "sentiment",
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
    "hatespeech"]

item_labels = [
    "Sentiment",
    "Respect",
    "Insult",
    "Humiliate",
    "Status",
    "Dehumanize",
    "Violence",
    "Genocide",
    "Attack/Defend",
    "Hate Speech"]

"""
Target columns
"""
# Column names for target groups
target_groups = [
    'target_race',
    'target_religion',
    'target_origin',
    'target_gender',
    'target_sexuality',
    'target_age',
    'target_disability',
    'target_politics']

target_labels = [group.split('_')[1].capitalize()
                 for group in target_groups]

# Targets race columns
target_race_cols = [
    'target_race_asian',
    'target_race_black',
    'target_race_latinx',
    'target_race_middle_eastern',
    'target_race_native_american',
    'target_race_pacific_islander',
    'target_race_white',
    'target_race_other']

# Targets religion columns
target_religion_cols = [
    'target_religion_atheist',
    'target_religion_buddhist',
    'target_religion_christian',
    'target_religion_hindu',
    'target_religion_jewish',
    'target_religion_mormon',
    'target_religion_muslim',
    'target_religion_other',]

# Targets national origin columns
target_origin_cols = [
    'target_origin_immigrant',
    'target_origin_migrant_worker',
    'target_origin_specific_country',
    'target_origin_undocumented',
    'target_origin_other']

# Targets gender column
target_gender_cols = [
    'target_gender_men',
    'target_gender_non_binary',
    'target_gender_transgender_men',
    'target_gender_transgender_unspecified',
    'target_gender_transgender_women',
    'target_gender_women',
    'target_gender_other']

# Targets sexuality column
target_sexuality_cols = [
    'target_sexuality_bisexual',
    'target_sexuality_gay',
    'target_sexuality_lesbian',
    'target_sexuality_straight',
    'target_sexuality_other']

# Targets age column
target_age_cols = [
    'target_age_children',
    'target_age_teenagers',
    'target_age_young_adults',
    'target_age_middle_aged',
    'target_age_seniors',
    'target_age_other']

# Targets disability column
target_disability_cols = [
    'target_disability_physical',
    'target_disability_cognitive',
    'target_disability_neurological',
    'target_disability_visually_impaired',
    'target_disability_hearing_impaired',
    'target_disability_unspecific',
    'target_disability_other']

# Targets politics column
target_politics_cols = [
    'target_politics_alt_right',
    'target_politics_communist',
    'target_politics_conservative',
    'target_politics_democrat',
    'target_politics_green_party',
    'target_politics_leftist',
    'target_politics_liberal',
    'target_politics_libertarian',
    'target_politics_republican',
    'target_politics_socialist',
    'target_politics_other']

# All targets
target_cols = target_race_cols + \
              target_religion_cols + \
              target_origin_cols + \
              target_gender_cols + \
              target_sexuality_cols + \
              target_age_cols + \
              target_disability_cols + \
              target_politics_cols

"""
Annotator columns
"""
# Annotator race
annotator_race_cols = [
    'annotator_race_asian',
    'annotator_race_black',
    'annotator_race_latinx',
    'annotator_race_middle_eastern',
    'annotator_race_native_american',
    'annotator_race_pacific_islander',
    'annotator_race_white',
    'annotator_race_other']


# Annotator gender
annotator_gender_cols = [
    'annotator_gender_men',
    'annotator_gender_women',
    'annotator_gender_non_binary',
    'annotator_gender_prefer_not_to_say',
    'annotator_gender_self_describe']


# Annotator transgender
annotator_trans_cols = [
    'annotator_transgender',
    'annotator_cisgender',
    'annotator_transgender_prefer_not_to_say']


# Annotator religion
annotator_religion_cols = [
    'annotator_religion_atheist',
    'annotator_religion_buddhist',
    'annotator_religion_christian',
    'annotator_religion_hindu',
    'annotator_religion_jewish',
    'annotator_religion_mormon',
    'annotator_religion_muslim',
    'annotator_religion_other']


# Annotator sexuality
annotator_sexuality_cols = [
    'annotator_sexuality_bisexual',
    'annotator_sexuality_gay',
    'annotator_sexuality_straight',
    'annotator_sexuality_other']


# Annotator ideology
annotator_ideology_cols = [
    'annotator_ideology_extremeley_conservative',
    'annotator_ideology_conservative',
    'annotator_ideology_slightly_conservative',
    'annotator_ideology_neutral',
    'annotator_ideology_slightly_liberal',
    'annotator_ideology_liberal',
    'annotator_ideology_extremeley_liberal',
    'annotator_ideology_no_opinion']


# Annotator education
annotator_education_cols = [
    'annotator_education_some_high_school',
    'annotator_education_high_school_grad',
    'annotator_education_some_college',
    'annotator_education_college_grad_aa',
    'annotator_education_college_grad_ba',
    'annotator_education_professional_degree',
    'annotator_education_masters',
    'annotator_education_phd']


# Annotator income
annotator_income_cols = [
    'annotator_income_<10k',
    'annotator_income_10k-50k',
    'annotator_income_50k-100k',
    'annotator_income_100k-200k',
    'annotator_income_>200k']
