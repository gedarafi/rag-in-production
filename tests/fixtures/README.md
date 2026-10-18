# Test fixtures

`anilist-mini.csv` is a tiny catalog in the AniList export format (the columns the recommender reads), with
ten invented anime, so the recommender can be built and deployed in seconds on LocalStack and in CI
(`just recommender-publish fixture`). None of its titles, characters or descriptions are real.
