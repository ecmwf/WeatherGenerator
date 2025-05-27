// Variable filler
// Variable filler

local fill_variables(vars) = {
  [k]: {
    description: vars[k][0],
    mean: vars[k][1],
    std: vars[k][2],
    tendency_mean: vars[k][3],
    tendency_std: vars[k][4],
    level: vars[k][5],
  }
  for k in std.objectFields(vars)
};

local fill_geometry(vars) = {
  "type": "Polygon",
  "coordinates": [
    [
      [vars[0], vars[2]],
      [vars[0], vars[3]],
      [vars[1], vars[3]],
      [vars[1], vars[2]],
      [vars[0], vars[2]]
    ]	  
  ]
};

local fill_assets(d) = {
  [d.dataset_name]: {
    title: d.dataset_name,
    type: d.type,
    roles: d.roles,
    version: d.version,
    notes: d.notes,
    locations: d.locations,
    size: d.size,
    inodes: d.inodes
  }
};

// Optional: create catalogue link
local dataset_entry_catalogue(ds, href_link) = {
  rel: "item",
  href: href_link + "/" + ds.filename,
  title: ds.title,
  type: "application/json"
};

// Create full STAC item for a dataset
local dataset_entry_fill(ds) = {
  type: "Feature",
  stac_version: "1.0.0",
  id: "weathergen.atmo." + ds.name,
  description: ds.description,
  unique_id: ds.unique_id,
  title: ds.title,
  keywords: ds.keywords,
  providers: ds.providers,
  start_datetime: ds.start_datetime,
  end_datetime: ds.end_datetime,
  "cube:variables": fill_variables(ds.variables),
  geometry: fill_geometry(ds.geometry), 
  bbox: [ds.geometry[0], ds.geometry[2], ds.geometry[1], ds.geometry[3]],
  stac_extensions: [
      "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
      "https://stac-extensions.github.io/alternate-assets/v1.2.0/schema.json",
      "https://stac-extensions.github.io/xarray-assets/v1.0.0/schema.json"
    ],
  assets: fill_assets(ds.dataset) 
};

{
  fill_variables: fill_variables,
  fill_geometry: fill_geometry,
  fill_assets: fill_assets,
  dataset_entry_catalogue: dataset_entry_catalogue,
  dataset_entry_fill: dataset_entry_fill,
}