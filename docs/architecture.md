# Novyrix Architecture

Browser
    |
    v
Novyrix API
    |
    v
Novyrix Engine
    |
    +-- Model Registry
    |
    +-- Memory
    |
    +-- Context
    |
    +-- File handling
    |
    v
Novyrix Runtime
    |
    v
Novyrix Model

The runtime is deliberately separated from the API so the model
implementation can change without rewriting the frontend.
