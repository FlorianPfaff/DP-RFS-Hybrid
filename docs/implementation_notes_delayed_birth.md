# Implementation Notes: Delayed Birth Learning

The first implementation step should be conservative and easy to test:

- keep immediate birth learning as the default behavior;
- add an opt-in delayed mode on `LabeledMultiBernoulliTracker`;
- in delayed mode, an accepted birth still creates a tentative track immediately;
- the DP birth atoms are not updated immediately;
- once the track reaches a configurable age and existence threshold, its current state is used as confirmed birth evidence.

This avoids changing the default demo behavior while creating the research hook needed for two-time-scale learning.
