
Git
#
We forked kawin from repo below
upstream  git@github.com:materialsgenomefoundation/kawin.git
might need to merge new commits from kawin main repo using command below:

git fetch upstream
git rebase upstream/main
#
 
Docker Commands                                                        
# Build                                                           
docker build -t kawin-api:latest .                                          
# Run                                                             
docker run -d -p 8000:8000 kawin-api:latest                                
# Test
curl http://localhost:8000/health 