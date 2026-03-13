import numpy as np
import torch

for epoch in range(cfg.epochs):
            self.train()
  
            # 2 For epochs
            # 3 Calculate the gradient by formula (7)
            
            # 4 Update the parameters by gradient descent (10)
            
            # 5 At the end of the iteration process, use the updated parameters a and b, and
            # train the coding part again with forward propagation to obtain self-expression
            # table for the layer.
            
            # 6 Implement the algorithm in the self-expression layer part of the last layer 2.
            
            

            optimizer.zero_grad()
            J.backward()
            optimizer.step()

      
            self.u_ = u.detach()
            self.omega_ = omega.detach()

            if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0 or epoch == cfg.epochs - 1):
                omega_sum = float(self.omega_.sum().item()) if self.omega_ is not None else float("nan")
                mode = "mid-only" if use_mid_only else "multilayer"
                print(
                    f"epoch {epoch+1:4d}/{cfg.epochs} [{mode}]  "
                    f"J={J.item():.4e}  J1={J1.item():.4e}  J2={J2.item():.4e}  J3={J3.item():.4e}  "
                    f"omega_sum={omega_sum:.4f}"
                )

        return self
