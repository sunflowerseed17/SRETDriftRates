library(brms)
library(tidyverse)    # for dplyr, stringr, etc.
library(data.table)   # for fast data loading

# Use sum-to-zero contrasts for factors
options(contrasts = c("contr.sum", "contr.poly"))

# Load the CSV data
df <- fread("data/SRET2019.csv") %>%
  # Remove trials with missing RT or response
  filter(!is.na(SERT.RT), !is.na(Response)) %>%
  # Create columns for model
  mutate(
    rt = SERT.RT / 1000,  # convert RT from ms to seconds
    choice = ifelse(Response %in% c(1, "yes", "Yes", "YES"), 1, 0),  # 1 = "yes", 0 = "no"
    Subject = as.factor(Subject),
    Valence = tolower(Valence),
    Circumplex = str_replace_all(tolower(Circumplex), " ", "")
  ) %>%
  # Keep only relevant Valence-Circumplex combinations:
  filter(
    (Valence == "positive" & Circumplex %in% c("highaffiliation", "highdominance")) |
    (Valence == "negative" & Circumplex %in% c("lowaffiliation", "lowdominance"))
  ) %>%
  # Recode factors with desired levels
  mutate(
    Valence = factor(Valence, levels = c("negative", "positive")),
    Circumplex = case_when(
      str_detect(Circumplex, "affiliation") ~ "affiliation",
      str_detect(Circumplex, "dominance") ~ "dominance"
    ),
    Circumplex = factor(Circumplex, levels = c("affiliation", "dominance"))
  )

# Step 2: Fit the hierarchical DDM (Wiener) model
model_vc <- brm(
  formula = bf(rt | dec(choice) ~ Valence * Circumplex + (1 | Subject)),
  data = df,
  family = wiener(),
  prior = c(
    prior(normal(0, 1), class = "b"),           # coefficients for drift
    prior(normal(0, 1), class = "Intercept"),   # drift intercept
    prior(normal(1.5, 1), class = "bs", lb = 0),# boundary separation
    prior(normal(0.3, 0.1), class = "ndt", lb = 0), # non-decision time
    prior(beta(2, 2), class = "bias")           # start-point bias
  ),
  chains = 4, iter = 4000, cores = 4,
  control = list(adapt_delta = 0.99, max_treedepth = 12)
)

# Step 3: Print model summary
print(summary(model_vc))

# Step 4: Plot marginal effects of Valence and Circumplex
plot_data <- conditional_effects(model_vc, effects = "Valence:Circumplex")
plot(plot_data, points = TRUE)

# Step 5: Extract subject-level absolute drift rates per condition

# Define labels for the 4 combinations:
condition_labels <- c("negative_affiliation", "negative_dominance", 
                      "positive_affiliation", "positive_dominance")

# Get fixed effect estimates
fixed_effects <- fixef(model_vc)
intercept     <- fixed_effects["Intercept", "Estimate"]
valence_eff   <- fixed_effects["Valence1", "Estimate"]         # effect of positive vs negative (contr.sum coded)
circ_eff      <- fixed_effects["Circumplex1", "Estimate"]      # effect of dominance vs affiliation
interaction_eff <- fixed_effects["Valence1:Circumplex1", "Estimate"]  # interaction effect

# Compute the fixed drift rate for each condition:
cond_fixed_effects <- tibble(
  Condition = condition_labels,
  FixedDrift = c(
    intercept,                                                # negative_affiliation (baseline in contr.sum coding context)
    intercept + circ_eff,                                     # negative_dominance
    intercept + valence_eff,                                  # positive_affiliation
    intercept + valence_eff + circ_eff + interaction_eff      # positive_dominance
  )
)

# Extract subject-level random intercepts for drift
subject_intercepts <- ranef(model_vc)$Subject[ , , "Intercept"][ , "Estimate"]
# `subject_intercepts` is a named vector indexed by Subject ID.

# Combine subject intercepts with fixed condition effects for each subject:
library(purrr)
subject_cond_df <- map_dfr(names(subject_intercepts), function(subj) {
  subj_intercept <- subject_intercepts[subj]
  tibble(
    Subject   = subj,
    Condition = condition_labels,
    Drift     = cond_fixed_effects$FixedDrift + subj_intercept
  )
})

# Merge in clinical symptom scores for each subject (assuming these were in the original data frame)
symptoms <- df %>%
  select(Subject, LSAS, SPIN, FPES, BFNE, BDI, RSES, `STAI-S`, `STAI-T`) %>%
  distinct()

subject_merged <- left_join(subject_cond_df, symptoms, by = "Subject")
head(subject_merged)

write.csv(subject_merged, "output/dm_absolute_drift_subjects.csv", row.names = FALSE)
