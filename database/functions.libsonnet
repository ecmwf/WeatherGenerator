// Variable filler


local check_unique_ids(datasets) = 
  local ids = [ds.unique_id for ds in datasets];
  local unique_ids = std.set(ids);
  if std.length(unique_ids) != std.length(ids) then
    error "Duplicate in unique IDs: " + std.join(", ", ids)
  else
    {};

local check_variable_lengths(vars) = 
  if std.objectHas(vars, 'means' ) && std.length(vars.names) != std.length(vars.means) then
    error "lengths of variables and means do not match"
  else if std.objectHas(vars, 'stds' ) && std.length(vars.names) != std.length(vars.stds) then
    error "lengths of variables and stds do not match"
  else if std.objectHas(vars, 'mins' ) && std.length(vars.names) != std.length(vars.mins) then
    error "length of variables and minimum do not match"
  else if std.objectHas(vars, 'maxs' ) && std.length(vars.names) != std.length(vars.maxs) then
    error "lengths of variables and maximum do not match"
  
  else if std.objectHas(vars, 'tendencies') && 
      std.length(vars.tendencies.means) > 0 && 
      std.length(vars.names) != std.length(vars.tendencies.means) then
      error "lengths of variables and tendencies (means) do not match"
  else if std.objectHas(vars, 'tendencies') && 
      std.length(vars.tendencies.stds) > 0 && 
      std.length(vars.names) != std.length(vars.tendencies.stds) then
      error "lengths of variables and tendencies (stds) do not match"
  else
    {};

  

local fill_variables(vars) = 
  check_variable_lengths(vars)
  +
  {
    [vars.names[k]]: {
      min : (if std.objectHas(vars, 'mins' ) then vars.mins[k] else "NA"),
      max : (if std.objectHas(vars, 'maxs' ) then vars.maxs[k] else "NA"),
      mean: (if std.objectHas(vars, 'means') then vars.means[k] else "NA"),
      std : (if std.objectHas(vars, 'stds' ) then vars.stds[k] else "NA"),
      tendency_mean: (
          if std.objectHas(vars, 'tendencies') && std.length(vars.tendencies.means) > 0
          then vars.tendencies.means[k]
          else "NA"
        ),
      tendency_std: (
        if std.objectHas(vars, 'tendencies') && std.length(vars.tendencies.stds) > 0
        then vars.tendencies.stds[k]
        else "NA"
      ),
    }
    for k in std.range(0, std.length(vars.names) - 1)
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
  fixed_timesteps: ds.fixed_timesteps, 
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
  check_unique_ids: check_unique_ids, 
  fill_variables: fill_variables,
  fill_geometry: fill_geometry,
  fill_assets: fill_assets,
  dataset_entry_catalogue: dataset_entry_catalogue,
  dataset_entry_fill: dataset_entry_fill,
}