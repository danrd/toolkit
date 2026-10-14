"""Pieces for pipelines built around a model: where the time goes (trace), blocks added to a module's prompt
(blocks), and the loop that asks again with the reasons the last answer was refused (refine). None of them knows
what the model is asked; the project gives what is specific to it as functions."""
