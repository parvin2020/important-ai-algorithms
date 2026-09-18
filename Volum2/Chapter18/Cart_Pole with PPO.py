import gymnasium as gym
from stable_baselines3 import PPO

# ==========================================
# بخش اول: آموزش عامل (Training Phase)
# ==========================================
print("---Start of Agent Training ---")

# ۱. ساخت محیط CartPole
train_env = gym.make("CartPole-v1")

# ۲. تعریف مدل PPO با معماری MlpPolicy
model = PPO("MlpPolicy", train_env, verbose=1)

# ۳. آموزش مدل در ۵۰,۰۰۰ گام زمان
model.learn(total_timesteps=5_000)

# ۴. ذخیره مدل آموزش‌دیده و بستن محیط آموزش
model.save("ppo_cartpole")
train_env.close()

print("--- Training Finished Successfully & Model Saved. --")


# ==========================================
# بخش دوم: اجرای عامل آموزش‌دیده (Evaluation / Test)
# ==========================================
print("--- Start of Running Trained Model ---")

# ۱. ساخت محیط مجدد همراه با نمایش گرافیکی (render_mode="human")
test_env = gym.make("CartPole-v1", render_mode="human")

# ۲. بارگذاری مدل ذخیره‌شده
loaded_model = PPO.load("ppo_cartpole")

# ۳. مقداردهی اولیه محیط
observation, info = test_env.reset()

# ۴. حلقه اجرای عامل در محیط
for step in range(100):
    # پیش‌بینی حرکت بعدی براساس وضعیت فعلی
    action, _ = loaded_model.predict(observation, deterministic=True)

    # اعمال حرکت در محیط و دریافت وضعیت جدید
    observation, reward, terminated, truncated, info = test_env.step(action)

    # اگر بازی تمام شد یا زمان آن به پایان رسید، محیط بازنشانی می‌شود
    if terminated or truncated:
        observation, info = test_env.reset()

# ۵. بستن پنجره محیط
test_env.close()
print("---Running Model Finished ---")