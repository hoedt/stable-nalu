
source('./_function_task_expand_name.r')

expand.name = function (df) {
  names = data.frame(name=unique(df$name))
  
  df.expand.name = names %>%
    rowwise() %>%
    mutate(
      model=revalue(extract.by.split(name, 1), model.full.to.short, warn_missing=FALSE),
      operation=revalue(extract.by.split(name, 2), operation.full.to.short, warn_missing=FALSE), # op
      
      regualizer.scaling = regualizer.get.type(extract.by.split(name, 3), 1), # rs
      
      regualizer.scaling.start=regualizer.scaling.get(extract.by.split(name, 4), 1),
      regualizer.scaling.end=regualizer.scaling.get(extract.by.split(name, 4), 2),
      
      regualizer=regualizer.get.part(extract.by.split(name, 5), 1),
      regualizer.z=regualizer.get.part(extract.by.split(name, 5), 2),
      regualizer.oob=regualizer.get.part(extract.by.split(name, 5), 3),
      regualizer.l2=regualizer.get.part(extract.by.split(name, 5), 4),
      
      interpolation.range=range.full.to.short(extract.by.split(name, 6)),
      extrapolation.range=range.full.to.short(extract.by.split(name, 7)),

      input.size=dataset.get.part(extract.by.split(name, 8), 1, 4),
      seq.length=dataset.get.part(extract.by.split(name, 8), 2, NA),
      subset.ratio=dataset.get.part(extract.by.split(name, 8), 3, NA),
      overlap.ratio=dataset.get.part(extract.by.split(name, 8), 4, NA),

      hidden.size=as.integer(substring(extract.by.split(name, 9, 'h2'), 2)),
      learning.rate=as.numeric(substring(extract.by.split(name, 11, 'lr-0.001'), 4)),
      batch.size=as.integer(substring(extract.by.split(name, 12), 2)),
      seed=as.integer(substring(extract.by.split(name, 13), 2)),
    )
  
  df.expand.name$name = as.factor(df.expand.name$name)
  df.expand.name$operation = factor(df.expand.name$operation, c('$\\times$', '$\\mathbin{/}$', '$+$', '$-$', '$\\sqrt{z}$', '$z^2$'))
  df.expand.name$model = as.factor(df.expand.name$model)
  df.expand.name$interpolation.range = as.factor(df.expand.name$interpolation.range)
  df.expand.name$extrapolation.range = as.factor(df.expand.name$extrapolation.range)
  
  #return(df.expand.name)
  return(merge(df, df.expand.name))
}
