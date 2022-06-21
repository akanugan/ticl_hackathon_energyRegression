# Data Format

One graph per trackster

- Nodes: Clusters
- Properties: 
    - position  -> vertices_{x,y,z}
    - energy    -> vertices_energy
    - time      -> to be added later

- One extra node for trackster information
    - position  -> Barycenter
    - energy    -> raw_energy
    - pt        -> raw_pt


To combine cluster nodes and trackster nodes they have to be padded 
```python
        self.ts_encode = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.lc_encode = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
```

Both tracksters (`ts`) and layer clusters (`lc`) would be encodedn the same way to then have the same dimensionality (`hidden_dim`). This allows them to be used together in subsequent convolutional layers. 