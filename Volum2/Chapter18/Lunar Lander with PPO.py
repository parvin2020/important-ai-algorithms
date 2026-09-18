import gymnasium as gym
from stable_baselines3 import PPO

# ==========================================
# ۱. آموزش عامل فرود روی ماه (LunarLander)
# ==========================================
print("--- Start of Training of Moon Lander ---")

# ساخت محیط فرود روی ماه
train_env = gym.make("LunarLander-v3")

# ساخت مدل PPO
model = PPO("MlpPolicy", train_env, verbose=1)

# آموزش مدل (۱۰۰ هزار گام برای یادگیری فرود نرم)
model.learn(total_timesteps=10_000)

# ذخیره مدل
model.save("ppo_lunar_lander")
train_env.close()

print("--- End of Training of Moon_Lander ---")


# ==========================================
# ۲. اجرای گرافیکی مدل آموزش‌دیده
# ==========================================
print("---Showing of Performance  ---")

# ساخت محیط با نمایش بصری
test_env = gym.make("LunarLander-v3", render_mode="human")

# بارگذاری مدل
loaded_model = PPO.load("ppo_lunar_lander")

# مقداردهی اولیه
observation, info = test_env.reset()

# اجرای ۵ اپیزود کامل فرود
for episode in range(5):
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # دریافت اکشن از مدل
        action, _ = loaded_model.predict(observation, deterministic=True)
        
        # اعمال اکشن (روشن کردن موتور چپ، راست یا اصلی)
        observation, reward, terminated, truncated, info = test_env.step(action)
    
    # ریست محیط برای اپیزود بعدی
    observation, info = test_env.reset()

test_env.close()
print("---End of Running ---")