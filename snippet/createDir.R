# # create saving dir: 
createDir = function(main.dir, sub.dir){
  if (dir.exists(file.path(main.dir, sub.dir))){
    return(saving.dir = file.path(main.dir, sub.dir))
  } else {
    return(saving.dir = dir.create(file.path(main.dir, sub.dir), F))
  }
}