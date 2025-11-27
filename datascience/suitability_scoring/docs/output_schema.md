# Outout schema for suitability scoring
This document explains the output schema from the suitability scoring. The output from the recommender system needs to include the exclusions handling output plus this output.

## Minimal Output Schema

*  `farm_id` *(int, required)* - ID from farms.
*  `timestamp_utc` *(string, ISO 8601)* - run timestamp.
*  `recommendations` *(list, required)* - List of recommended species.
    *  `species_id` *(int, required)* - ID from species.
    *  `species_name` *(string, required)* - Scientific name of species.
    *  `species_common_name` *(string, required)* - convenience label.
    *  `score_mcda` *(float, \[0,1], required)* - final weighted suitability score.
    *  `rank_overall` *(int)* - rank across all species for this farm (1 = best).
    *  `key_reasons` *(string, required)* - List of reasons for each scored feature. 

**JSON example**

```json
{
  "farm_id": 1,
  "timestamp_utc": "2025-11-20T22:35:00Z",
  "recommendations": [
    {
      "species_id": 10,
      "species_name": "Acacia",
      "species_common_name": "Acacia",
      "score_mcda": 0.82,
      "rank_overall": 1,
      "key_reasons": ["rainfall:ideal", "soil:match", "temp: below optimal"],
    },
    {
      "species_id": 2,
      "species_name": "Eucalyptus",
      "score_mcda": 0.76,
      "key_reasons": ["rainfall:below optimal", "soil:match", "ph:ideal"]
    }
  ]
}
```