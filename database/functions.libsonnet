// Variable filler

local fill_variables(vars) = {
  [k]: {
    min: vars[k][0],
    max: vars[k][1],
    mean: vars[k][2],
    std: vars[k][3],
    tendency_mean: if std.length(vars[k]) > 4 then vars[k][4] else "NA",
    tendency_std: if std.length(vars[k]) > 5 then vars[k][5] else "NA",
  }
  for k in std.objectFields(vars)
};

local fill_properties(ds) = {
  name: ds.name,
  description: ds.description,
  unique_id: ds.unique_id,
  title: ds.title,
  start_datetime: ds.start_datetime,
  end_datetime: ds.end_datetime,
  keywords: ds.keywords,
  providers: ds.providers,
  variables: fill_variables(ds.variables),
  frequency: ds.frequency,
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

local fill_assets(ds) = {
  [ds.dataset_name]: {
    title: ds.dataset_name,
    href: ds.dataset_name,
    type: ds.type,
    roles: ds.roles,
    description: ds.description,
    locations: ds.locations,
    size: ds.size,
    inodes: ds.inodes
  }
};

// Optional: create catalogue link
local dataset_entry_catalogue(ds, href_link) = {
  rel: "child",
  href: href_link + "/" + ds.filename,
  title: ds.title,
  type: "application/json"
};

// Create full STAC item for a dataset
local dataset_entry_fill(ds) = {
  "type": "Feature",
  "stac_version": "1.0.0",
  "id": "weathergen.atmo." + ds.name,
   "properties": fill_properties(ds),
  "geometry": fill_geometry(ds.geometry), 
  "bbox": [ds.geometry[0], ds.geometry[2], ds.geometry[1], ds.geometry[3]],
  "stac_extensions": [
      "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
      "https://stac-extensions.github.io/alternate-assets/v1.2.0/schema.json",
      "https://stac-extensions.github.io/xarray-assets/v1.0.0/schema.json"
    ],
  "assets": fill_assets(ds.dataset),
};

{
  fill_variables: fill_variables,
  fill_geometry: fill_geometry,
  fill_assets: fill_assets,
  dataset_entry_catalogue: dataset_entry_catalogue,
  dataset_entry_fill: dataset_entry_fill,
}